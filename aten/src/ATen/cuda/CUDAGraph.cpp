#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Logging.h>
#include <c10/util/ScopeExit.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <optional>

namespace at::cuda {

// To support stream capture across multiple threads, we use a global
// hashmap mapping cuda stream capture IDs to CUDAGraph objects. This
// was originally a thread_local std::stack<CUDAGraph*>, but that was
// not acceptable since stream capture does span threads in certain
// circumstances (in particular, during autograd).
static std::mutex _currently_capturing_graphs_mutex;
static ska::flat_hash_map<CaptureId_t, CUDAGraph*> _currently_capturing_graphs;

struct CaptureState {
  CUDAGraph* graph;
  bool ending{false};
  std::vector<CUDAGraph::CaptureEpilogue> epilogues;
  std::vector<CUDAGraph::CaptureReplayCallback> replay_callbacks;
  std::vector<std::shared_ptr<void>> resources;
  std::exception_ptr error;
};

// Keep capture epilogues and their resources out of CUDAGraph itself so adding
// this internal facility does not change the size of the exported C++ class.
static std::mutex _capture_state_mutex;
static ska::flat_hash_map<CaptureId_t, CaptureState> _capture_states;
static ska::flat_hash_map<CUDAGraph*, std::vector<std::shared_ptr<void>>>
    _retained_capture_resources;

struct ReplayCallbackState {
  std::vector<CUDAGraph::CaptureReplayCallback> callbacks;
};

static ska::flat_hash_map<CUDAGraph*, std::shared_ptr<ReplayCallbackState>>
    _capture_replay_callbacks;
static std::atomic<size_t> _graphs_with_replay_callbacks{0};
static std::atomic<uint64_t> _replay_callback_registry_generation{0};

struct ReplayCallbackCache {
  CUDAGraph* graph{nullptr};
  uint64_t registry_generation{0};
  std::weak_ptr<const ReplayCallbackState> state;
};

static thread_local ReplayCallbackCache _replay_callback_cache;

static std::shared_ptr<const ReplayCallbackState> get_replay_callback_state(
    CUDAGraph* graph) {
  // Keep ordinary CUDAGraph replay free of registry locking. Callback-bearing
  // graphs are rare, and capture/reset mutations invalidate the thread-local
  // cache used by repeated replays.
  if (_graphs_with_replay_callbacks.load(std::memory_order_acquire) == 0) {
    return nullptr;
  }
  const auto registry_generation =
      _replay_callback_registry_generation.load(std::memory_order_acquire);
  if (_replay_callback_cache.graph == graph &&
      _replay_callback_cache.registry_generation == registry_generation) {
    return _replay_callback_cache.state.lock();
  }

  std::shared_ptr<const ReplayCallbackState> state;
  {
    std::lock_guard<std::mutex> lock(_capture_state_mutex);
    auto it = _capture_replay_callbacks.find(graph);
    if (it != _capture_replay_callbacks.end()) {
      state = it->second;
    }
  }
  _replay_callback_cache = {graph, registry_generation, state};
  return state;
}

#if defined(USE_ROCM)
// Returns true when at least one CUDAGraph capture is currently active in this
// process. Uses the same mutex-protected capture map as capture lifecycle
// bookkeeping.
bool is_graph_capture_active() {
  std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
  return !_currently_capturing_graphs.empty();
}
#endif // defined(USE_ROCM)

CUDAGraph* get_graph_from_capture_id(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
  auto it = _currently_capturing_graphs.find(capture_id);
  if (it != _currently_capturing_graphs.end()) {
    return it->second;
  }
  return nullptr;
}

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return at::cuda::MemPool::graph_pool_handle();
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

CUDAGraph::CUDAGraph(bool keep_graph)
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()),
    keep_graph_(keep_graph) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

bool CUDAGraph::has_retained_pool(MempoolId_t pool) const {
  for (const auto& retained_pool : retained_mempool_ids_) {
    if (retained_pool == pool) {
      return true;
    }
  }
  return false;
}

void CUDAGraph::record_retained_pool(MempoolId_t pool) {
  if (!has_retained_pool(pool)) {
    retained_mempool_ids_.push_back(pool);
  }
}

void CUDAGraph::retain_pool(MempoolId_t pool) {
  TORCH_CHECK(
      capture_id_ != 0 && !capture_ended_,
      "CUDAGraph::retain_pool may only be called during capture.");
  TORCH_CHECK(
      pool.first != 0 || pool.second != 0,
      "CUDAGraph::retain_pool expected a non-default memory pool.");
  if (has_retained_pool(pool)) {
    return;
  }
  c10::cuda::CUDACachingAllocator::createOrIncrefPool(capture_dev_, pool);
  record_retained_pool(pool);
}

void CUDAGraph::begin_capture_state(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto [it, inserted] =
      _capture_states.emplace(capture_id, CaptureState{this});
  TORCH_INTERNAL_ASSERT(
      inserted,
      "Duplicate CUDAGraph capture state for capture ID ",
      capture_id);
}

void CUDAGraph::register_capture_epilogue(
    CaptureId_t capture_id,
    CaptureEpilogue epilogue) {
  TORCH_CHECK(epilogue, "CUDAGraph capture epilogue must be callable.");
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto it = _capture_states.find(capture_id);
  TORCH_CHECK(
      it != _capture_states.end() && it->second.graph == this &&
          !it->second.ending,
      "Cannot register an epilogue for inactive CUDA Graph capture ",
      capture_id,
      ".");
  it->second.epilogues.push_back(std::move(epilogue));
}

void CUDAGraph::register_capture_replay_callback(
    CaptureId_t capture_id,
    CaptureReplayCallback callback) {
  TORCH_CHECK(callback, "CUDAGraph replay callback must be callable.");
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto it = _capture_states.find(capture_id);
  TORCH_CHECK(
      it != _capture_states.end() && it->second.graph == this &&
          !it->second.ending,
      "Cannot register a replay callback for inactive CUDA Graph capture ",
      capture_id,
      ".");
  it->second.replay_callbacks.push_back(std::move(callback));
}

void CUDAGraph::retain_capture_resource(
    CaptureId_t capture_id,
    std::shared_ptr<void> resource) {
  TORCH_CHECK(resource, "CUDAGraph cannot retain a null capture resource.");
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto it = _capture_states.find(capture_id);
  TORCH_CHECK(
      it != _capture_states.end() && it->second.graph == this &&
          !it->second.ending,
      "Cannot retain a resource for inactive CUDA Graph capture ",
      capture_id,
      ".");
  it->second.resources.push_back(std::move(resource));
}

CUDAGraph::CaptureEndState CUDAGraph::begin_capture_end(
    CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto it = _capture_states.find(capture_id);
  if (it == _capture_states.end() || it->second.graph != this) {
    return {};
  }
  it->second.ending = true;
  return {std::move(it->second.epilogues), it->second.error};
}

void CUDAGraph::record_capture_error(
    CaptureId_t capture_id,
    std::exception_ptr error) {
  TORCH_INTERNAL_ASSERT(error);
  std::lock_guard<std::mutex> lock(_capture_state_mutex);
  auto it = _capture_states.find(capture_id);
  if (it != _capture_states.end() && it->second.graph == this &&
      !it->second.error) {
    it->second.error = std::move(error);
  }
}

void CUDAGraph::finish_capture_end(CaptureId_t capture_id, bool success) {
  std::vector<CaptureReplayCallback> callbacks_to_release;
  std::shared_ptr<ReplayCallbackState> callback_state_to_release;
  std::vector<std::shared_ptr<void>> resources_to_release;
  {
    std::lock_guard<std::mutex> lock(_capture_state_mutex);
    auto it = _capture_states.find(capture_id);
    if (it == _capture_states.end() || it->second.graph != this) {
      return;
    }
    if (success) {
      if (!it->second.replay_callbacks.empty()) {
        auto callbacks_it = _capture_replay_callbacks.find(this);
        if (callbacks_it == _capture_replay_callbacks.end()) {
          callbacks_it = _capture_replay_callbacks
                             .emplace(
                                 this, std::make_shared<ReplayCallbackState>())
                             .first;
          _graphs_with_replay_callbacks.fetch_add(1, std::memory_order_release);
        }
        for (auto& callback : it->second.replay_callbacks) {
          callbacks_it->second->callbacks.push_back(std::move(callback));
        }
        _replay_callback_registry_generation.fetch_add(
            1, std::memory_order_release);
      }
      for (auto& resource : it->second.resources) {
        _retained_capture_resources[this].push_back(std::move(resource));
      }
    } else {
      callbacks_to_release = std::move(it->second.replay_callbacks);
      resources_to_release = std::move(it->second.resources);
    }
    _capture_states.erase(it);
    if (!success && capture_id == capture_id_) {
      auto callbacks_it = _capture_replay_callbacks.find(this);
      if (callbacks_it != _capture_replay_callbacks.end()) {
        callback_state_to_release = std::move(callbacks_it->second);
        _capture_replay_callbacks.erase(callbacks_it);
        _graphs_with_replay_callbacks.fetch_sub(1, std::memory_order_release);
        _replay_callback_registry_generation.fetch_add(
            1, std::memory_order_release);
      }
      auto retained_it = _retained_capture_resources.find(this);
      if (retained_it != _retained_capture_resources.end()) {
        for (auto& resource : retained_it->second) {
          resources_to_release.push_back(std::move(resource));
        }
        _retained_capture_resources.erase(retained_it);
      }
    }
  }
}

template <>
std::function<bool(cudaStream_t)> CUDAGraph::create_allocate_filter<cudaStream_t>() const {
  return [this](cudaStream_t stream) {
    auto capture_id_opt = c10::cuda::captureIdMayInitCtx(stream);
    return capture_id_opt.has_value() && capture_id_opt.value() == capture_id_;
  };
}

template <>
std::function<bool(c10::Stream)> CUDAGraph::create_allocate_filter<c10::Stream>() const {
  return [this](c10::Stream stream) {
    cudaStream_t cuda_stream = CUDAStream(CUDAStream::UNCHECKED, stream);
    auto capture_id_opt = c10::cuda::captureIdMayInitCtx(cuda_stream);
    return capture_id_opt.has_value() && capture_id_opt.value() == capture_id_;
  };
}

void CUDAGraph::capture_begin(MempoolId_t pool/*={0,0}*/, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  capture_mode_ = capture_mode;

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

#if defined(USE_ROCM)
  // hipBLASLt handles are per-(device, stream) on ROCm and lazily created.
  // Ensure the handle for the intended capture stream exists before
  // capture begins, because hipblasLtCreate performs internal allocations
  // that are not allowed once stream capture is active.
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
    (void)at::cuda::getCurrentCUDABlasLtHandle();
  }
#endif

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = at::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_allocate_filter<cudaStream_t>());
  record_retained_pool(mempool_id_);

  at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, create_allocate_filter<c10::Stream>());

  // The pool is now acquired and being recorded to. Track this so reset() can
  // release it even if the capture fails before capture_end() completes.
  allocated_pool_ = true;
  capturing_to_pool_ = true;

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));
  c10::cuda::CUDACachingAllocator::markCaptureBegin(capture_dev_);

  auto capture_id_opt = c10::cuda::captureIdMayInitCtx(stream);
  TORCH_INTERNAL_ASSERT(capture_id_opt.has_value(),
      "Stream should be actively capturing after cudaStreamBeginCapture");
  capture_id_ = capture_id_opt.value();
  begin_capture_state(capture_id_);

  {
    std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(capture_id_, this);
  }
}

// capture_end is split so callers can run work on the captured cudaGraph_t
// (e.g. read its id, dump it, transform it) in the window between the end of
// capture and finalization, when graph_ is live for both keep_graph modes.
// capture_end_post finalizes by destroying the template for keep_graph=false;
// instantiation is driven separately (by capture_end for C++ callers, or by the
// Python wrapper) so it has a single entry point. capture_end runs the whole
// sequence for callers that don't need the window.
void CUDAGraph::capture_end_pre() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream.stream() == capture_stream_.stream(),
              "Capture must end on the same stream it began on.");

  auto capture_end_state = begin_capture_end(capture_id_);
  std::exception_ptr epilogue_error = capture_end_state.error;
  for (auto& epilogue : capture_end_state.epilogues) {
    try {
      epilogue(stream);
    } catch (...) {
      if (!epilogue_error) {
        epilogue_error = std::current_exception();
      }
    }
  }

  // Capture is over once cudaStreamEndCapture returns (success or failure).
  // Clear bookkeeping before propagating the return status so watchdog-side
  // checks cannot observe stale "capture active" state on error paths.
  cudaError_t endCaptureErr = cudaStreamEndCapture(capture_stream_, &graph_);
  c10::cuda::CUDACachingAllocator::markCaptureEnd(capture_dev_);
  bool erased_capture = false;
  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    erased_capture = _currently_capturing_graphs.erase(capture_id_) == 1;
  }

  // End pool allocation before checking the capture error. This ensures
  // captures_underway is cleaned up even if cudaStreamEndCapture failed
  // (e.g. due to an illegal operation during capture). These calls are
  // safe regardless of whether the capture succeeded — they simply
  // remove the pool routing entry added by beginAllocateToPool.
  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
  // Allocation recording has stopped (even if endCaptureErr is a failure), so
  // reset() must not end the pool again.
  capturing_to_pool_ = false;

  const bool capture_succeeded =
      endCaptureErr == cudaSuccess && graph_ != nullptr && !epilogue_error;
  if (!capture_succeeded) {
    if (graph_ != nullptr) {
      C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
      graph_ = nullptr;
    }
    finish_capture_end(capture_id_, false);
    TORCH_INTERNAL_ASSERT(
        erased_capture, "capture_end() called before capture_begin().");
    if (epilogue_error) {
      std::rethrow_exception(epilogue_error);
    }
    AT_CUDA_CHECK(endCaptureErr);
    TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
  }

  finish_capture_end(capture_id_, true);
  TORCH_INTERNAL_ASSERT(
      erased_capture, "capture_end() called before capture_begin().");
  AT_CUDA_CHECK(endCaptureErr);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increment] :
       captured_generator_states_) {
    wholegraph_increment = generator_state->capture_epilogue(capture_id_);
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;
}

void CUDAGraph::capture_end_post() {
  // Destroy-only: when keep_graph=false the template is not retained. The graph
  // must already be instantiated (capture_end and the Python wrapper instantiate
  // before calling this).
  if (!keep_graph_ && has_graph_) {
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
}

void CUDAGraph::capture_end() {
  capture_end_pre();
  if (!keep_graph_) {
    instantiate();
  }
  capture_end_post();
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if !defined(USE_ROCM)
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority));
#else
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
#endif
  has_graph_exec_ = true;
}

void CUDAGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::replay without a preceding successful capture.");
  // Instantiating on demand is handled by the Python replay() wrapper (which
  // can do so for keep_graph=true). At this level the exec graph must exist.
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay before the graph was instantiated; "
              "call instantiate() first.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  auto replay_callback_state = get_replay_callback_state(this);

  for (auto& [generator_state, wholegraph_increment] :
       captured_generator_states_) {
    generator_state->replay_prologue(capture_id_, wholegraph_increment);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
  std::exception_ptr callback_error;
  if (replay_callback_state) {
    for (const auto& callback : replay_callback_state->callbacks) {
      try {
        callback();
      } catch (...) {
        if (!callback_error) {
          callback_error = std::current_exception();
        }
      }
    }
  }
  if (callback_error) {
    std::rethrow_exception(callback_error);
  }
}

void CUDAGraph::enable_debug_mode() {
  // Debug mode just retains the template after capture so it can be inspected
  // (e.g. dumped); that is exactly what keep_graph does. Unify on keep_graph_
  // rather than a second flag. dot dumping itself lives in Python now
  // (torch.cuda.CUDAGraph.debug_dump via cuda.bindings).
  keep_graph_ = true;
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(has_graph_,
      "No cudaGraph_t is available: either capture_end() has not been called, "
      "or the underlying cudaGraph_t was destroyed (keep_graph=false, and "
      "capture has been finalized).");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

void CUDAGraph::reset() {
  // Remove global ownership first so even a later allocator-cleanup exception
  // cannot leave dangling CUDAGraph pointers behind. Keep the payloads in
  // locals, declared before the scope guard, so graphs are destroyed before
  // arbitrary retained objects and epilogue closures are released.
  std::vector<CaptureEpilogue> capture_epilogues_to_release;
  std::vector<CaptureReplayCallback> capture_callbacks_to_release;
  std::shared_ptr<ReplayCallbackState> callback_state_to_release;
  std::vector<std::shared_ptr<void>> capture_resources_to_release;
  auto destroy_graphs_on_exit = c10::make_scope_exit([this] {
    if (has_graph_) {
      C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
      graph_ = nullptr;
      has_graph_ = false;
    }
    if (has_graph_exec_) {
      C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
      graph_exec_ = nullptr;
      has_graph_exec_ = false;
    }
  });
  {
    std::lock_guard<std::mutex> lock(_capture_state_mutex);
    for (auto it = _capture_states.begin(); it != _capture_states.end();) {
      if (it->second.graph != this) {
        ++it;
        continue;
      }
      for (auto& epilogue : it->second.epilogues) {
        capture_epilogues_to_release.push_back(std::move(epilogue));
      }
      for (auto& callback : it->second.replay_callbacks) {
        capture_callbacks_to_release.push_back(std::move(callback));
      }
      for (auto& resource : it->second.resources) {
        capture_resources_to_release.push_back(std::move(resource));
      }
      it = _capture_states.erase(it);
    }
    auto callbacks_it = _capture_replay_callbacks.find(this);
    if (callbacks_it != _capture_replay_callbacks.end()) {
      callback_state_to_release = std::move(callbacks_it->second);
      _capture_replay_callbacks.erase(callbacks_it);
      _graphs_with_replay_callbacks.fetch_sub(1, std::memory_order_release);
      _replay_callback_registry_generation.fetch_add(
          1, std::memory_order_release);
    }
    auto retained_it = _retained_capture_resources.find(this);
    if (retained_it != _retained_capture_resources.end()) {
      for (auto& resource : retained_it->second) {
        capture_resources_to_release.push_back(std::move(resource));
      }
      _retained_capture_resources.erase(retained_it);
    }
  }

  // These checks warn instead of throwing: reset() is called from the
  // destructor, and at least one CI build refuses to compile with a throwing
  // destructor. Resource cleanup lives here in C++ so it runs on garbage
  // collection regardless of Python state; the Python __del__ on
  // torch.cuda.CUDAGraph only tears down bookkeeping and intentionally does not
  // call reset().
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.

  // See Note [RNG state tensor lifetime and recordStream] in
  // CUDAGeneratorImpl.cpp — recordStream in setup_for_replay ensures the
  // allocator won't recycle these tensors until in-flight replays finish.
  if (capture_id_ != 0) {
    for (auto& [generator_state, wholegraph_increment] : captured_generator_states_) {
      generator_state->remove_capture_state(capture_id_);
    }
  }
  captured_generator_states_.clear();

  if (capture_id_ != 0) {
    std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.erase(capture_id_);
    capture_id_ = 0;
  }

  if (allocated_pool_) {
    if (capturing_to_pool_) {
      // Capture was abandoned before capture_end() ran, so the allocator is
      // still routing allocations to this pool. Stop that before releasing so
      // the pool is left in a consistent, freeable state.
      c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
      at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
      capturing_to_pool_ = false;
    }

    // Clean up cuBLAS workspaces allocated on the capture stream, otherwise live allocations prevent
    // private pool cleanup
    clearCublasWorkspacesForStream(capture_stream_.stream());

    // notifyCaptureDestroy may throw. How should we handle this?
    for (const auto& pool : retained_mempool_ids_) {
      c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, pool);
    }
    retained_mempool_ids_.clear();
    at::getHostAllocator(at::kCUDA)->release_pool(mempool_id_);
    capture_ended_ = false;
    allocated_pool_ = false;
  }
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  while (!conditional_node_raw_streams_.empty()) {
    conditional_node_raw_streams_.pop();
  }
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

std::vector<MempoolId_t> CUDAGraph::pools() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pools() without a preceding successful capture.");
  return retained_mempool_ids_;
}

CUDAGraph::~CUDAGraph() {
  try {
    reset();

// There are recent HIP changes where hipGraphExecDestroy doesn't immediately free memory.
// They wait for next sync point in order to free the memory, this is to ensure that all
// hipGraphLaunch are finished before we release any memory. This feature was enabled in rocm6.2.
// We need to ensure all async operations finish before deleting the object.
#if defined(USE_ROCM)
    if (capture_dev_ != UNDEFINED_DEVICE) // check if capture_dev_ contains the real device id
    {
      AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to clean up CUDAGraph: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Failed to clean up CUDAGraph: unknown exception";
  }
}

CUDAGraph* CUDAGraph::get_currently_capturing_graph() {
  std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
  auto capture_id_opt = c10::cuda::currentStreamCaptureIdMayInitCtx();
  TORCH_CHECK(
      capture_id_opt.has_value(),
      "The current stream is not currently capturing.");
  TORCH_CHECK(
      _currently_capturing_graphs.count(capture_id_opt.value()),
      "get_currently_capturing_graph() can be used only between capture_begin() and capture_end(). Did you use a stream without making it depend upon the original stream used for capture?");
  return _currently_capturing_graphs.at(capture_id_opt.value());
}

void CUDAGraph::begin_capture_to_if_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  begin_capture_to_conditional_node(
      scalar_cuda_pred_tensor, cudaGraphCondTypeIf);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

void CUDAGraph::begin_capture_to_while_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  begin_capture_to_conditional_node(
      scalar_cuda_pred_tensor, cudaGraphCondTypeWhile);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return;
#endif
}

#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
void CUDAGraph::begin_capture_to_conditional_node(
    const at::Tensor& scalar_cuda_pred_tensor,
    cudaGraphConditionalNodeType conditional_type) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This CUDAGraph instance already owns a captured graph.");

  TORCH_CHECK(!c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::graph_capture_record_stream_reuse(), "'graph_capture_record_stream_reuse:True' allocator config does not work with conditional control flow in a cuda graph today. See issue #175001 for updates");

  cudaStreamCaptureStatus status{};
  cudaGraph_t currently_capturing_graph{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(), &status, nullptr, &currently_capturing_graph));
  TORCH_CHECK(
      status == cudaStreamCaptureStatusActive,
      "capture_begin() must be called before begin_capture_to_conditional_node()");
  cudaGraphConditionalHandle handle{};
  AT_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, currently_capturing_graph, 0, 0));

  set_conditional_handle(handle, scalar_cuda_pred_tensor);

  const cudaGraphNode_t* dependencies{};
  const cudaGraphEdgeData* dependency_edges{};
  size_t num_dependencies = 0;
#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &dependency_edges,
      &num_dependencies));
#else
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
      getCurrentCUDAStream(),
      &status,
      nullptr,
      &currently_capturing_graph,
      &dependencies,
      &dependency_edges,
      &num_dependencies
  ));
#endif
  TORCH_CHECK(status == cudaStreamCaptureStatusActive);

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = conditional_type;
  params.conditional.size = 1;

  cudaGraphNode_t cond_node{};
#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaGraphAddNode(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      dependency_edges,
      num_dependencies,
      &params));
#else
  AT_CUDA_CHECK(cudaGraphAddNode_v2(
      &cond_node,
      currently_capturing_graph,
      dependencies,
      dependency_edges,
      num_dependencies,
      &params));
#endif
  cudaGraph_t conditional_node_child_graph = params.conditional.phGraph_out[0];

#if CUDA_VERSION >= 13000
  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
getCurrentCUDAStream(), &cond_node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
  AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies_v2(
getCurrentCUDAStream(), &cond_node, nullptr, 1, cudaStreamSetCaptureDependencies));
#endif

  cudaStream_t raw_child_stream{};
  AT_CUDA_CHECK(cudaStreamCreateWithFlags(
      &raw_child_stream, cudaStreamNonBlocking));
  CUDAStream child_stream =
      getStreamFromExternal(raw_child_stream, capture_dev_);
  conditional_node_raw_streams_.emplace(raw_child_stream);
  conditional_graph_capture_ids_.push(0);
  conditional_node_handles_.push(handle);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, create_child_allocate_filter());
  auto filter = create_child_allocate_filter();
  at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
    return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
  });

  AT_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
      child_stream,
      conditional_node_child_graph,
      nullptr,
      nullptr,
      0,
      capture_mode_));
  c10::cuda::CUDACachingAllocator::markCaptureBegin(capture_dev_);

  auto child_capture_id_opt = c10::cuda::captureIdMayInitCtx(child_stream);
  TORCH_INTERNAL_ASSERT(child_capture_id_opt.has_value(),
      "Child stream should be actively capturing after cudaStreamBeginCaptureToGraph");
  conditional_graph_capture_ids_.top() = child_capture_id_opt.value();
  begin_capture_state(conditional_graph_capture_ids_.top());

  conditional_node_streams_.emplace(child_stream);

  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    _currently_capturing_graphs.emplace(
        conditional_graph_capture_ids_.top(), this);
  }
}
#endif // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)

void CUDAGraph::end_capture_to_conditional_node() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_INTERNAL_ASSERT(
      !conditional_graph_capture_ids_.empty(),
      "Missing capture ID for conditional node.");

  CaptureId_t child_capture_id = conditional_graph_capture_ids_.top();
  bool rng_or_generators_changed = false;
  for (const auto& [generator_state, wholegraph_increment] :
       captured_generator_states_) {
    if (generator_state->get_capture_state(child_capture_id) != nullptr) {
      rng_or_generators_changed = true;
      break;
    }
  }

  CUDAStream stream = conditional_node_streams_.top().current_stream();
  auto capture_end_state = begin_capture_end(child_capture_id);
  std::exception_ptr child_capture_error = capture_end_state.error;
  for (auto& epilogue : capture_end_state.epilogues) {
    try {
      epilogue(stream);
    } catch (...) {
      if (!child_capture_error) {
        child_capture_error = std::current_exception();
      }
    }
  }

  cudaError_t endCaptureErr = cudaStreamEndCapture(stream.stream(), nullptr);
  c10::cuda::CUDACachingAllocator::markCaptureEnd(capture_dev_);

  bool erased_capture = false;
  {
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    erased_capture = _currently_capturing_graphs.erase(child_capture_id) == 1;
  }

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
  at::getHostAllocator(at::kCUDA)->end_allocate_to_pool(mempool_id_);

  conditional_node_streams_.pop();
  conditional_graph_capture_ids_.pop();
  conditional_node_handles_.pop();

  TORCH_INTERNAL_ASSERT(!conditional_node_raw_streams_.empty());
  conditional_node_raw_streams_.pop();

  if (conditional_graph_capture_ids_.empty()) {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_allocate_filter<cudaStream_t>());
    at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, create_allocate_filter<c10::Stream>());
  } else {
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        capture_dev_, mempool_id_, create_child_allocate_filter());
    auto filter = create_child_allocate_filter();
    at::getHostAllocator(at::kCUDA)->begin_allocate_to_pool(mempool_id_, [filter](c10::Stream stream) {
      return filter(CUDAStream(CUDAStream::UNCHECKED, stream));
    });
  }
  constexpr const char* rng_with_conditional_nodes_error =
      "RNG within data-dependent conditional nodes is not supported yet.";
  if (!child_capture_error && endCaptureErr != cudaSuccess) {
    try {
      AT_CUDA_CHECK(endCaptureErr);
    } catch (...) {
      child_capture_error = std::current_exception();
    }
  }
  if (!child_capture_error && rng_or_generators_changed) {
    try {
      TORCH_CHECK(false, rng_with_conditional_nodes_error);
    } catch (...) {
      child_capture_error = std::current_exception();
    }
  }
  finish_capture_end(child_capture_id, endCaptureErr == cudaSuccess);
  TORCH_INTERNAL_ASSERT(
      erased_capture, "capture_end() called before capture_begin().");
  if (child_capture_error) {
    record_capture_error(capture_id_, child_capture_error);
    std::rethrow_exception(child_capture_error);
  }

#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
#endif
}

void CUDAGraph::set_conditional_handle_for_current_node(
    const at::Tensor& scalar_cuda_pred_tensor) {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  TORCH_INTERNAL_ASSERT(
      !conditional_node_handles_.empty(),
      "No active CUDA graph conditional node.");
  set_conditional_handle(
      conditional_node_handles_.top(), scalar_cuda_pred_tensor);
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
#endif
}

std::function<bool(cudaStream_t)> CUDAGraph::create_child_allocate_filter() {
#if !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  return [&current_capture_id = conditional_graph_capture_ids_.top()](cudaStream_t stream) {
      auto capture_id_opt = c10::cuda::captureIdMayInitCtx(stream);
      return capture_id_opt.has_value() && capture_id_opt.value() == current_capture_id;
  };
#else // !defined(USE_ROCM) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  AT_ERROR(
      __func__,
      " CUDA Graphs conditional nodes are not supported for cuda version < 12.4");
  return std::function<bool(cudaStream_t)>();
#endif
}


} // namespace at::cuda
