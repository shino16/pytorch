#include <ATen/cuda/CUDAGreenContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGreenContextMacros.h>

#if HAS_CUDA_GREEN_CONTEXT()
#include <c10/cuda/driver_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#else
// Suppress unused private field warnings as this class is not supposed to be called
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-private-field")
#endif

namespace at::cuda {

uint32_t get_num_locality_domains(std::optional<uint32_t> device_id) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  int driver_version;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 13040, "cuda driver too old to use green context localization!");
  if (!device_id.has_value()) {
    device_id = at::cuda::current_device();
  }
  CUdevice device;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id.value()));
  int count = 0;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
    &count, CU_DEVICE_ATTRIBUTE_MEMORY_NODE_COUNT, device));
  return static_cast<uint32_t>(count);
#else
  TORCH_CHECK(false, "get_num_locality_domains is only supported on CUDA 13.4+!");
  return 0;
#endif
}

std::unique_ptr<GreenContext> GreenContext::create(
  std::optional<uint32_t> device_id,
  std::optional<uint32_t> num_sms,
  std::optional<int32_t> workqueue_scope,
  std::optional<uint32_t> workqueue_concurrency_limit,
  std::optional<int32_t> locality_domain_id) {
#if HAS_CUDA_GREEN_CONTEXT()
int driver_version;
C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
TORCH_CHECK(
    driver_version >= 12080, "cuda driver too old to use green context!");
if (workqueue_scope.has_value() || workqueue_concurrency_limit.has_value()) {
  TORCH_CHECK(
      driver_version >= 13010, "cuda driver too old to use workqueue configuration!");
}
if (locality_domain_id.has_value()) {
  TORCH_CHECK(
      driver_version >= 13040, "cuda driver too old to use green context localization!");
}
if (!device_id.has_value()) {
  device_id = at::cuda::current_device();
}
return std::unique_ptr<GreenContext>(new GreenContext(
    device_id.value(), num_sms, workqueue_scope, workqueue_concurrency_limit, locality_domain_id));
#else
TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
return nullptr;
#endif
}

uint32_t GreenContext::max_workqueue_concurrency(
  std::optional<uint32_t> device_id) {
#if HAS_CUDA_WORKQUEUE_SUPPORT()
int driver_version;
C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
TORCH_CHECK(
    driver_version >= 13010, "cuda driver too old to use workqueue configuration!");
if (!device_id.has_value()) {
  device_id = at::cuda::current_device();
}
CUdevice device;
C10_CUDA_DRIVER_CHECK(
    c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id.value()));
CUdevResource wq_resource;
C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
    device, &wq_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
return wq_resource.wqConfig.wqConcurrencyLimit;
#else
TORCH_CHECK(false, "Workqueue configuration requires CUDA 13.1+!");
return 0;
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
  return c10::cuda::getDefaultCUDAStream();
#endif
}

int32_t GreenContext::device_id() const {
  return device_id_;
}

int32_t GreenContext::locality_domain_id() const {
  return locality_domain_id_;
}

bool GreenContext::has_locality_domain() const {
  return locality_domain_id_ >= 0;
}

int32_t GreenContext::num_locality_domains() const {
  return num_locality_domains_;
}

GreenContext::GreenContext(
    uint32_t device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope,
    std::optional<uint32_t> workqueue_concurrency_limit,
    std::optional<int32_t> locality_domain_id) {
#if HAS_CUDA_GREEN_CONTEXT()
  TORCH_CHECK(
      num_sms.has_value() || workqueue_scope.has_value() || locality_domain_id.has_value(),
      "At least one of num_sms or workqueue_scope or locality_domain_id must be specified");
  TORCH_CHECK(
      !locality_domain_id.has_value() || !num_sms.has_value(),
      "locality_domain_id and num_sms cannot be specified together");
  TORCH_CHECK(
      !workqueue_concurrency_limit.has_value() || workqueue_scope.has_value(),
      "workqueue_concurrency_limit requires workqueue_scope to be set");

  CUcontext pctx = nullptr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&pctx));
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN(
        "Attempted to create a green context but"
        " there was no primary context! Creating a primary context...");

    C10_CUDA_CHECK(cudaFree(nullptr));
  }

  CUdevice device;
  device_id_ = device_id;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));

  std::vector<CUdevResource> resources;

  // --- SM resource ---
  if (num_sms.has_value()) {
    CUdevResource sm_resource;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
        device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));

    TORCH_CHECK(
        *num_sms > 0 && *num_sms <= sm_resource.sm.smCount,
        "Invalid number of SMs requested for green context: ",
        *num_sms,
        " (device has ",
        sm_resource.sm.smCount,
        " SMs)");

    // Split resources
    std::vector<CUdevResource> split_result(1);
    unsigned int nb_groups = 1;
    CUdevResource remaining;

    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuDevSmResourceSplitByCount_(
            split_result.data(),
            &nb_groups,
            &sm_resource,
            &remaining,
            0, // default flags
            *num_sms));
    TORCH_CHECK(nb_groups == 1, "Failed to create single SM resource group");
    resources.push_back(split_result[0]);
  }

  // --- Workqueue config resource ---
  if (workqueue_scope.has_value()) {
#if HAS_CUDA_WORKQUEUE_SUPPORT()
    CUdevResource wq_resource{};
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
        device, &wq_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));

    wq_resource.wqConfig.sharingScope =
        static_cast<CUdevWorkqueueConfigScope>(*workqueue_scope);
    if (workqueue_concurrency_limit.has_value()) {
      wq_resource.wqConfig.wqConcurrencyLimit = *workqueue_concurrency_limit;
    }
    resources.push_back(wq_resource);
#else
    TORCH_CHECK(
        false,
        "Workqueue configuration for green contexts requires CUDA 13.1+!");
#endif
  }

  if (locality_domain_id.has_value()) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
    // cache the localized SM resources for the locality domains, per device id
    static std::unordered_map<uint32_t, std::vector<CUdevResource>> localizedSms;
    if (localizedSms.find(device_id) == localizedSms.end()) {
      // first get normal SM resource
      CUdevResource sm_resource;
      C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
          device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));

      // get the number of locality domains
      int count = 0;
      C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetAttribute_(
        &count, CU_DEVICE_ATTRIBUTE_MEMORY_NODE_COUNT, device));
      TORCH_CHECK(count > 0, "No locality domains found on device!");
      auto& deviceLocalizedSms = localizedSms[device_id];
      deviceLocalizedSms.resize(count);

      // TODO: in the future, we might want to support custom coscheduled SM count
      // and remainder resource

      // set up params for resource splitting
      std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(count);
      for (int i = 0; i < count; i++) {
        std::memset(params.data() + i, 0, sizeof(params[i]));
        params[i].smCount = 0; // Use discovery mode of the API to derive SM count
        params[i].coscheduledSmCount = 2; // The minimum cluster capability: 2 SMs
        params[i].flags = CU_DEV_SM_RESOURCE_GROUP_MEMORY_NODE_ID;
        params[i].memoryNodeId = static_cast<unsigned char>(i);
      }
      // split the SM resource into the locality domains
      C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDevSmResourceSplit_(
        deviceLocalizedSms.data(), count, &sm_resource, /*remainder resource*/nullptr, /*flags*/0, params.data()));
    }
    auto const& deviceLocalizedSms = localizedSms[device_id];
    auto count = static_cast<int32_t>(deviceLocalizedSms.size());
    TORCH_CHECK(
        locality_domain_id.value() >= 0 && locality_domain_id.value() < count,
        "Invalid locality domain ID!");
    resources.push_back(deviceLocalizedSms[locality_domain_id.value()]);
    locality_domain_id_ = locality_domain_id.value();
    num_locality_domains_ = count;
#else
    TORCH_CHECK(
        false, "Green Context localization is only supported on CUDA 13.4+!");
#endif
  }

  // Generate resource descriptor
  CUdevResourceDesc desc;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevResourceGenerateDesc_(
          &desc,
          resources.data(),
          static_cast<unsigned int>(resources.size())));

  // Create green context
  // CU_GREEN_CTX_DEFAULT_STREAM is required per docs:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxCreate_(
      &green_ctx_, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  // Convert to regular context
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuCtxFromGreenCtx_(&context_, green_ctx_));
  TORCH_CHECK(context_, "Green ctx conversion to regular ctx failed!");
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

// Implement move operations
#if HAS_CUDA_GREEN_CONTEXT()
GreenContext::GreenContext(GreenContext&& other) noexcept
    : device_id_(std::exchange(other.device_id_, -1)),
      locality_domain_id_(std::exchange(other.locality_domain_id_, -1)),
      num_locality_domains_(std::exchange(other.num_locality_domains_, 0)),
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
    locality_domain_id_ = std::exchange(other.locality_domain_id_, -1);
    num_locality_domains_ = std::exchange(other.num_locality_domains_, 0);
    green_ctx_ = std::exchange(other.green_ctx_, nullptr);
    context_ = std::exchange(other.context_, nullptr);
    parent_stream_ = std::exchange(other.parent_stream_, nullptr);
    curr_stream_idx_.exchange(other.curr_stream_idx_);
    std::swap(this->green_ctx_streams_, other.green_ctx_streams_);
  }
  return *this;
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
  return *this;
#endif
}

} // namespace at::cuda

#if HAS_CUDA_GREEN_CONTEXT() == 0
C10_DIAGNOSTIC_POP_AND_IGNORED_IF_DEFINED()
#endif
