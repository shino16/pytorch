#include <ATen/cuda/CUDAGreenContext.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#define HAS_CUDA_GREEN_CONTEXT() 1
#else
#define HAS_CUDA_GREEN_CONTEXT() 0
// Suppress unused private field warnings as this class is not supposed to be called
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-private-field")
#endif

#if HAS_CUDA_GREEN_CONTEXT() == 1 && CUDA_VERSION >= 13040
#define HAS_CUDA_GREEN_CONTEXT_LOCALIZATION() 1
#else
#define HAS_CUDA_GREEN_CONTEXT_LOCALIZATION() 0
#endif

namespace at::cuda {

namespace {

// helper function to create a green context from a resource
void create_green_ctx_from_resource(
    CUgreenCtx& green_ctx,
    CUcontext& context,
    CUdevice device,
    CUdevResource& resource) {
#if HAS_CUDA_GREEN_CONTEXT()
  CUdevResourceDesc desc;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevResourceGenerateDesc_(
          &desc, &resource, 1));
  // CU_GREEN_CTX_DEFAULT_STREAM is required per docs:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxCreate_(
      &green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxFromGreenCtx_(
      &context, green_ctx));
  TORCH_CHECK(context, "Green ctx conversion to regular ctx failed!");
#endif
}

}  // namespace

uint32_t get_num_memory_nodes(uint32_t device_id) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  CUdevice device;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));
  int count = 0;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
    &count, CU_DEVICE_ATTRIBUTE_MEMORY_NODE_COUNT, device));
  return static_cast<uint32_t>(count);
#else
  TORCH_CHECK(false, "get_num_memory_nodes is only supported on CUDA 13.4+!");
  return 0;
#endif
}

GreenContext::GreenContext(int32_t device_id, int32_t memory_node_id, int32_t num_memory_nodes)
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
    : device_id_(device_id), memory_node_id_(memory_node_id), num_memory_nodes_(num_memory_nodes) { }
#else
  {
    TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  }
#endif

GreenContext::GreenContext(uint32_t device_id, uint32_t num_sms) {
#if HAS_CUDA_GREEN_CONTEXT()
  int driver_version;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 12080, "cuda driver too old to use green context!");
  CUcontext pctx = nullptr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&pctx));
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN(
        "Attempted to create a green context but"
        " there was no primary context! Creating a primary context...");

    cudaFree(nullptr);
  }

  CUdevice device;
  device_id_ = device_id;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));

  // Get device resources
  CUdevResource device_resource;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
      device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));

  TORCH_CHECK(
      num_sms > 0 && num_sms <= device_resource.sm.smCount,
      "Invalid number of SMs requested for green context: ",
      num_sms,
      " (device has ",
      device_resource.sm.smCount,
      " SMs)");

  // Split resources
  std::vector<CUdevResource> result(1);
  auto result_data = result.data();
  unsigned int nb_groups = 1;
  CUdevResource remaining;

  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevSmResourceSplitByCount_(
          result_data,
          &nb_groups,
          &device_resource,
          &remaining,
          0, // default flags
          num_sms));

  TORCH_CHECK(nb_groups == 1, "Failed to create single resource group");

  create_green_ctx_from_resource(green_ctx_, context_, device, result_data[0]);
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

std::unique_ptr<GreenContext> GreenContext::create(
  uint32_t num_sms,
  std::optional<uint32_t> device_id) {
#if HAS_CUDA_GREEN_CONTEXT()
  if (!device_id.has_value()) {
    device_id = at::cuda::current_device();
  }
  return std::unique_ptr<GreenContext>(new GreenContext(device_id.value(), num_sms));
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

std::vector<std::unique_ptr<GreenContext>> GreenContext::create_localized(
    std::optional<uint32_t> device_id) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  int driver_version = 0;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 13040, "cuda driver too old to use localized green contexts!");

  if (!device_id.has_value()) {
    device_id = at::cuda::current_device();
  }
  CUdevResource smResources;
  CUdevice device;
  C10_CUDA_DRIVER_CHECK(
    c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id.value()));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
    device, &smResources, CU_DEV_RESOURCE_TYPE_SM));

  int count = 0;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
    &count, CU_DEVICE_ATTRIBUTE_MEMORY_NODE_COUNT, device));
  TORCH_CHECK(count > 0, "No memory nodes found on device!");

  std::vector<CUdevResource> localizedSms(count);
  std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(count);
  for (int i = 0; i < count; i++) {
      std::memset(params.data() + i, 0, sizeof(params[i]));
      params[i].smCount = 0; // Use discovery mode of the API to derive SM count
      params[i].coscheduledSmCount = 2; // The minimum cluster capability: 2 SMs
      params[i].flags = CU_DEV_SM_RESOURCE_GROUP_MEMORY_NODE_ID;
      params[i].memoryNodeId = static_cast<unsigned char>(i);
  }
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDevSmResourceSplit_(
    localizedSms.data(), count, &smResources, /*remainder resource*/nullptr, /*flags*/0, params.data()));

  std::vector<std::unique_ptr<GreenContext>> green_contexts(count);
  for (int i = 0; i < count; i++) {
    green_contexts[i] = std::unique_ptr<GreenContext>(new GreenContext(device_id.value(), i, count));
    green_contexts[i]->device_id_ = device_id.value();
    green_contexts[i]->memory_node_id_ = i;
    green_contexts[i]->num_memory_nodes_ = count;
    create_green_ctx_from_resource(
      green_contexts[i]->green_ctx_, green_contexts[i]->context_, device, localizedSms[i]);
  }
  return green_contexts;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
#endif
}

// Implement move operations
#if HAS_CUDA_GREEN_CONTEXT()
GreenContext::GreenContext(GreenContext&& other) noexcept
    : device_id_(std::exchange(other.device_id_, -1)),
      memory_node_id_(std::exchange(other.memory_node_id_, -1)),
      num_memory_nodes_(std::exchange(other.num_memory_nodes_, 0)),
      green_ctx_(std::exchange(other.green_ctx_, nullptr)),
      context_(std::exchange(other.context_, nullptr)),
      parent_stream_(std::exchange(other.parent_stream_, nullptr)) {
  curr_stream_idx_.exchange(other.curr_stream_idx_);
  std::swap(this->green_ctx_streams_, other.green_ctx_streams_);
}
#else
GreenContext::GreenContext(GreenContext&& other) noexcept {
  TORCH_CHECK(false, "Green Context move constructor is only supported on CUDA 12.8+!");
}
#endif

void GreenContext::destroy_resources() noexcept {
#if HAS_CUDA_GREEN_CONTEXT()
  // avoid throwing exceptions on destruction
  // note: if curr_stream_idx_ was never updated, loop doesn't run
  for (int i = std::min(curr_stream_idx_.load(), kStreamPerGreenContextPool - 1); i >= 0;
       i--) {
    if (!green_ctx_streams_[i]) continue;
    auto err = c10::cuda::DriverAPI::get()->cuStreamDestroy_(green_ctx_streams_[i]);
    if (err != CUDA_SUCCESS) {
      TORCH_WARN(
          "Failed to destroy green context side stream ",
          i,
          " with error code ",
          static_cast<int>(err));
    }
  }
  if (green_ctx_) {
    auto err = c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_);
    if (err != CUDA_SUCCESS) {
      TORCH_WARN(
          "Failed to destroy green context with error code ",
          static_cast<int>(err));
    }
  }
#endif
}

GreenContext& GreenContext::operator=(GreenContext&& other) noexcept {
#if HAS_CUDA_GREEN_CONTEXT()
  if (this != &other) {
    // Clean up current resources
    if (green_ctx_) {
      CUcontext current = nullptr;
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
      if (current == context_) {
        TORCH_CHECK(
            false,
            "attempting to overwrite current green ctx "
            "when it is active!");
      }
      destroy_resources();
    }

    // Take ownership of other's resources
    device_id_ = std::exchange(other.device_id_, -1);
    memory_node_id_ = std::exchange(other.memory_node_id_, -1);
    num_memory_nodes_ = std::exchange(other.num_memory_nodes_, 0);
    green_ctx_ = std::exchange(other.green_ctx_, nullptr);
    context_ = std::exchange(other.context_, nullptr);
    parent_stream_ = std::exchange(other.parent_stream_, nullptr);
    curr_stream_idx_.exchange(other.curr_stream_idx_);
    std::swap(this->green_ctx_streams_, other.green_ctx_streams_);
  }
  return *this;
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

GreenContext::~GreenContext() noexcept {
#if HAS_CUDA_GREEN_CONTEXT()
  destroy_resources();
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

// Make this context current
void GreenContext::setContext(bool block_current_stream) {
#if HAS_CUDA_GREEN_CONTEXT()
  auto current_stream = c10::cuda::getCurrentCUDAStream();
  parent_stream_ = current_stream.stream();

  at::cuda::CUDAEvent ev;
  if (block_current_stream) {
    ev.record(current_stream);
  }

    CUcontext current = nullptr;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
    if (!current) {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxSetCurrent_(context_));
    } else {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxPushCurrent_(context_));
    }
    // setContext API uses default stream
    // see GreenContext::Stream() for side-stream creation
    // note: importantly, this relies on the fact that the default stream
    // in c10 is always the NULL / 0 stream : this stream has a special meaning
    // for both the primary and green contexts. In case we push a green context,
    // the default stream will be the green context's main stream, which is
    // always non-blocking unlike the primary context's default stream.
    auto green_ctx_stream = c10::cuda::getDefaultCUDAStream();
    if (block_current_stream) {
      ev.block(green_ctx_stream);
    }
    c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream(green_ctx_stream));
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

void GreenContext::popContext(bool block_parent_stream) {
#if HAS_CUDA_GREEN_CONTEXT()
  // see above note about stream being hardcoded to the default stream
  at::cuda::CUDAEvent ev;
  if (block_parent_stream) {
    ev.record(c10::cuda::getCurrentCUDAStream());
  }
  CUcontext popped;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuCtxPopCurrent_(&popped));
  TORCH_INTERNAL_ASSERT(
      popped == context_, "expected popped context to be the current ctx");

  auto parent_stream = c10::cuda::getStreamFromExternal(parent_stream_, device_id_);
  if (block_parent_stream) {
    ev.block(parent_stream);
  }
  c10::cuda::setCurrentCUDAStream(parent_stream);
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

CUDAStream GreenContext::Stream() {
#if HAS_CUDA_GREEN_CONTEXT()
  curr_stream_idx_++;
  auto idx = curr_stream_idx_ % kStreamPerGreenContextPool;
  if (curr_stream_idx_ < kStreamPerGreenContextPool) {
    CUstream green_ctx_side_stream;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxStreamCreate_(
        &green_ctx_side_stream, green_ctx_, CU_STREAM_NON_BLOCKING, 0));
    // note: green_ctx_streams_ is private and thus inaccessible outside
    // of this class. There should not be any static instances of GreenContext,
    // thus we can simply destroy these streams in a destructor,
    // see also GreenContext::destroy_resources().
    green_ctx_streams_[idx] = green_ctx_side_stream;
    return c10::cuda::getStreamFromExternal(green_ctx_side_stream, device_id_);
  }
  return c10::cuda::getStreamFromExternal(green_ctx_streams_[idx], device_id_);
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

int32_t GreenContext::device_id() const {
  return device_id_;
}

int32_t GreenContext::memory_node_id() const {
  return memory_node_id_;
}

bool GreenContext::has_memory_node() const {
  return memory_node_id_ >= 0;
}

int32_t GreenContext::num_memory_nodes() const {
  return num_memory_nodes_;
}

#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
LocalizedGreenContextAllocator::LocalizedGreenContextAllocator(
    GreenContext* green_context)
    : torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator(
        [green_context](size_t size, int device, cudaStream_t stream) {
          return allocate_on_green_context(
              size, device, stream, green_context);
        },
        [green_context](
            void* ptr, size_t size, int device, cudaStream_t stream) {
          free_on_green_context(ptr, size, device, stream, green_context);
        }) {}
#else
LocalizedGreenContextAllocator::LocalizedGreenContextAllocator(GreenContext* green_context) {
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
}
#endif

void* LocalizedGreenContextAllocator::allocate_on_green_context(
    size_t size,
    int device,
    cudaStream_t stream,
    GreenContext* green_context) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  TORCH_INTERNAL_ASSERT(green_context, "GreenContext allocator requires a context.");
  TORCH_CHECK(device == green_context->device_id(),
    "Device mismatch. Allocator device: ", device,
    ", GreenContext device: ", green_context->device_id());
  CUgreenCtx stream_ctx;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuStreamGetGreenCtx_(
    stream, &stream_ctx));
  TORCH_CHECK(stream_ctx == green_context->green_ctx_,
    "Green context mismatch. Allocator stream ctx: ", stream_ctx,
    ", GreenContext green ctx: ", green_context->green_ctx_);
  CUdeviceptr ptr = 0;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE;
  prop.location.localized.deviceId = device;
  prop.location.localized.memoryNodeId = green_context->memory_node_id();
  size_t granularity = 1;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t nodePaddedSize = (size + (granularity - 1)) & ~(granularity - 1);
  CUmemGenericAllocationHandle handle;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemCreate_(
    &handle, nodePaddedSize, &prop, /*flags*/0));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
    &ptr, nodePaddedSize, granularity, /*addr hint*/0, /*flags*/0));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
    ptr, nodePaddedSize, /*offset*/0, handle, /*flags*/0));
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemSetAccess_(
    ptr, nodePaddedSize, &desc, /*count [#descriptors]*/1));

  return reinterpret_cast<void*>(ptr);
#endif
}

void LocalizedGreenContextAllocator::free_on_green_context(
    void* ptr,
    size_t size,
    int device,
    cudaStream_t stream,
    GreenContext* green_context) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  TORCH_INTERNAL_ASSERT(green_context, "GreenContext allocator requires a context.");
  TORCH_CHECK(device == green_context->device_id(),
    "Device mismatch. Allocator device: ", device,
    ", GreenContext device: ", green_context->device_id());
  CUgreenCtx stream_ctx;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuStreamGetGreenCtx_(
    stream, &stream_ctx));
  // while freeing, if we are using the default stream, we might not get
  // the expected context as it is associated with the primary context as well.
  // So we allow the default stream here, but in this case, we need to
  // push and pop the green context context as needed.
  TORCH_CHECK(stream_ctx == green_context->green_ctx_ || stream == nullptr,
    "Green context mismatch. Allocator stream ctx: ", stream_ctx,
    ", GreenContext green ctx: ", green_context->green_ctx_);
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE;
  prop.location.localized.deviceId = device;
  prop.location.localized.memoryNodeId = green_context->memory_node_id();
  size_t granularity = 1;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t paddedSize = (size + (granularity - 1)) & ~(granularity - 1);

  CUmemGenericAllocationHandle handle;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemRetainAllocationHandle_(
    &handle, ptr));

  if (stream_ctx != green_context->green_ctx_) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxPushCurrent_(
      green_context->context_));
  }
  CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr);
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemUnmap_(
    devPtr, paddedSize));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressFree_(
    devPtr, paddedSize));
  // Two releases: one to undo cuMemRetainAllocationHandle, one for the
  // original cuMemCreate.
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemRelease_(handle));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemRelease_(handle));
  if (stream_ctx != green_context->green_ctx_) {
    CUcontext popped;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxPopCurrent_(&popped));
    TORCH_INTERNAL_ASSERT(popped == green_context->context_,
      "expected popped context to be the current green ctx");
  }
#endif
}

std::vector<std::unique_ptr<LocalizedGreenContextMemPool>>
LocalizedGreenContextMemPool::create_node_pools(
    std::vector<GreenContext*> const& green_contexts,
    bool alloc_in_order) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  size_t first_idx = green_contexts.size();
  for (size_t i = 0; i < green_contexts.size(); i++) {
    TORCH_CHECK(green_contexts[i]->has_memory_node(),
      "GreenContext localized MemPool requires localized green context.");
    if (green_contexts[i]->memory_node_id() == 0) {
      first_idx = i;
      break;
    }
  }
  TORCH_CHECK(
      first_idx < green_contexts.size(),
      "create_node_pools requires a green context with memory_node_id 0");
  auto first_pool = std::unique_ptr<LocalizedGreenContextMemPool>(
    new LocalizedGreenContextMemPool(
      green_contexts[first_idx], alloc_in_order, nullptr));
  LocalizedGreenContextMemPool* first_pool_ptr = first_pool.get();
  std::vector<std::unique_ptr<LocalizedGreenContextMemPool>> node_pools;
  node_pools.reserve(green_contexts.size());
  for (size_t i = 0; i < green_contexts.size(); i++) {
    if (i == first_idx) {
      node_pools.push_back(std::move(first_pool));
    } else {
      node_pools.push_back(std::unique_ptr<LocalizedGreenContextMemPool>(
        new LocalizedGreenContextMemPool(
          green_contexts[i], alloc_in_order, first_pool_ptr)));
    }
  }
  return node_pools;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
#endif
}

LocalizedGreenContextMemPool::LocalizedGreenContextMemPool(
    GreenContext* green_context,
    bool alloc_in_order,
    LocalizedGreenContextMemPool* first_pool)
    : LocalizedGreenContextAllocatorHolder{
          std::make_shared<LocalizedGreenContextAllocator>(green_context)},
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
      MemPool(allocator, /*is_user_created=*/true, /*use_on_oom=*/false,
        /*no_split=*/false,
        /*on_begin_allocate=*/
        [this](int device) {
          TORCH_CHECK(
            green_context_->device_id() == device,
            "GreenContext MemPool device mismatch. MemPool device: ",
            device,
            ", GreenContext device: ",
            green_context_->device_id());
          if (is_alloc_in_order()) {
            if (green_context_->memory_node_id() == 0) {
              first_pool_->main_event_.record(c10::cuda::getCurrentCUDAStream());
            }
            green_context_->setContext(/*block_current_stream=*/false);
            first_pool_->main_event_.block(c10::cuda::getCurrentCUDAStream());
          } else {
            green_context_->setContext(/*block_current_stream=*/true);
          }
        },
        /*on_end_allocate=*/
        [this](int device) {
          TORCH_CHECK(
            green_context_->device_id() == device,
            "GreenContext MemPool device mismatch. MemPool device: ",
            device,
            ", GreenContext device: ",
            green_context_->device_id());
          if (is_alloc_in_order()) {
            first_pool_->node_events_[green_context_->memory_node_id()].record(
              c10::cuda::getCurrentCUDAStream());
            green_context_->popContext(/*block_parent_stream=*/false);
            auto last_id =
                first_pool_->green_context()->num_memory_nodes() - 1;
            if (green_context_->memory_node_id() == last_id) {
              for (auto& node_event : first_pool_->node_events_) {
                node_event.block(c10::cuda::getCurrentCUDAStream());
              }
            }
          } else {
            green_context_->popContext(/*block_parent_stream=*/true);
          }
        }),
#else
      MemPool(),
#endif
      green_context_(green_context),
      alloc_in_order_(alloc_in_order),
      main_event_(),
      node_events_(),
      first_pool_(first_pool == nullptr ? this : first_pool) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  TORCH_CHECK(green_context_, "GreenContext MemPool requires a context.");
  TORCH_CHECK(green_context->has_memory_node(),
    "GreenContext localized MemPool requires localized green context.");
  if (green_context->num_memory_nodes() <= 1) {
    TORCH_WARN_ONCE("GreenContext localized MemPool used with single memory node.");
  }
  if (first_pool_ == this) {
    node_events_.resize(green_context->num_memory_nodes());
  }
#endif
}

GreenContext* LocalizedGreenContextMemPool::green_context() const {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  return green_context_;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return nullptr;
#endif
}

bool LocalizedGreenContextMemPool::is_alloc_in_order() const {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  return alloc_in_order_;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return false;
#endif
}

void LocalizedGreenContextMemPool::set_alloc_in_order(bool alloc_in_order) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  alloc_in_order_ = alloc_in_order;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
#endif
}

// ---------------------------------------------------------------------------
// LocalizedAllocator (NativeAllocatorBackend for localized / uGPU allocation)
// ---------------------------------------------------------------------------
// Always used through NativeAllocatorBackendWrapper, which owns the deleter
// dispatch via its PtrMetadata map.  LocalizedAllocator::get_deleter() should
// never be called by the caching allocator (the wrapper's get_deleter() is
// used instead).

void* LocalizedAllocator::allocate(
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  int driver_version = 0;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(driver_version >= 13040, "cuda driver too old to use localized allocator!");

  CUdevice dev;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&dev, device));
  int count = 0;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
    &count, CU_DEVICE_ATTRIBUTE_MEMORY_NODE_COUNT, dev));
  TORCH_CHECK(count > 0 && count <= 2,
    "Currently the localized allocator only supports 2 memory nodes.");
  TORCH_CHECK(size % count == 0, "Size ", size,
    " must be a multiple of the number of memory nodes: ", count);

  TORCH_INTERNAL_ASSERT(count > 0 && count <= MAX_MEMORY_NODES,
    "Invalid number of memory nodes: ", count);

  // Create the memory allocation on uGPU0 and uGPU1 (two halves).
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_MEMORY_NODE;
  prop.location.localized.deviceId = device;
  prop.location.localized.memoryNodeId = 0;
  size_t granularity = 1;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  for (int i = 1; i < count; i++) {
    prop.location.localized.memoryNodeId = i;
    size_t granularity_i = 1;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
      &granularity_i, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    TORCH_INTERNAL_ASSERT(granularity_i == granularity,
      "granularity must be the same for all memory nodes");
  }

  // Each cuMemCreateLocalized size must be a multiple of granularity (driver API
  // requirement). So round each node's memory up to granularity.
  // in order to guarantee support for any C++ data type, we first ensure that
  // the node size is a multiple of 16 bytes.
  TORCH_INTERNAL_ASSERT(granularity % MIN_ALIGNMENT == 0,
    "granularity must be a multiple of MIN_ALIGN");
  size_t nodeSize = size / count;
  nodeSize = (nodeSize + MIN_ALIGNMENT - 1) & ~(MIN_ALIGNMENT - 1);
  size_t nodePaddedSize = (nodeSize + (granularity - 1)) & ~(granularity - 1);
  size_t paddedSize = nodePaddedSize * count;
  TORCH_INTERNAL_ASSERT(nodePaddedSize >= nodeSize,
    "nodePaddedSize must be greater or equal to nodeSize");

  PtrMetadata metadata{};
  metadata.allocator = this;
  metadata.nodePaddedSize = nodePaddedSize;
  metadata.nodeSize = nodeSize;
  metadata.memoryNodeCount = count;

  for (int i = 0; i < count; i++) {
    prop.location.localized.memoryNodeId = i;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemCreate_(
      &metadata.handles[i],
      nodePaddedSize,
      &prop,
      /*flags*/0));
  }

  CUdeviceptr devPtr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
    &devPtr, paddedSize, granularity, /*addr hint*/0, /*flags*/0));
  // cuMemMap requires ptr and size to be multiples of allocation granularity.
  // Thus, we will center any allocation around the granularity boundary.
  // Note that this does not work in case we have more than 2 memory nodes.
  for (int i = 0; i < count; i++) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
      devPtr + i * nodePaddedSize, nodePaddedSize, /*offset*/0, metadata.handles[i], /*flags*/0));
  }

  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemSetAccess_(
    devPtr, paddedSize, &desc, /*count [#descriptors]*/1));

  // here we center the output pointer around the granularity boundary.
  auto* ptr = reinterpret_cast<void*>(devPtr + (nodePaddedSize - nodeSize));
  {
    std::lock_guard<std::mutex> lock(ptr_mutex_());
    ptr_to_metadata_()[ptr] = metadata;
  }

  return ptr;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return nullptr;
#endif
}

void LocalizedAllocator::free(void* ptr) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  PtrMetadata metadata;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(ptr_mutex_());
    auto it = ptr_to_metadata_().find(ptr);
    if (it != ptr_to_metadata_().end()) {
      metadata = it->second;
      ptr_to_metadata_().erase(it);
      found = true;
    }
  }
  if (!found) return;
  free_(ptr, metadata);
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
#endif
}

void LocalizedAllocator::free_(void* ptr, PtrMetadata const& metadata) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(ptr) -
    (metadata.nodePaddedSize - metadata.nodeSize);
  for (int i = 0; i < metadata.memoryNodeCount; i++) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemUnmap_(
      devPtr + i * metadata.nodePaddedSize, metadata.nodePaddedSize));
  }
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressFree_(
    devPtr, metadata.nodePaddedSize * metadata.memoryNodeCount));
  for (int i = 0; i < metadata.memoryNodeCount; i++) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemRelease_(metadata.handles[i]));
  }
#endif
}

void LocalizedAllocator::static_deleter(void* ptr) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  LocalizedAllocator::PtrMetadata metadata;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(LocalizedAllocator::ptr_mutex_());
    auto it = LocalizedAllocator::ptr_to_metadata_().find(ptr);
    if (it != LocalizedAllocator::ptr_to_metadata_().end()) {
      metadata = it->second;
      LocalizedAllocator::ptr_to_metadata_().erase(it);
      found = true;
    }
  }
  if (found && metadata.allocator) {
    metadata.allocator->free_(ptr, metadata);
  }
#endif
}

c10::DeleterFnPtr LocalizedAllocator::get_deleter() const {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  return &static_deleter;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return nullptr;
#endif
}

std::string LocalizedAllocator::name() const {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  return "localized";
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return "";
#endif
}

} // namespace at::cuda
