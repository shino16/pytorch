#include <ATen/cuda/LocalizedGreenContextMemPool.h>
#include <ATen/cuda/CUDAGreenContextMacros.h>

#if HAS_CUDA_GREEN_CONTEXT()
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#endif

namespace at::cuda {
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
LocalizedGreenContextAllocator::LocalizedGreenContextAllocator(GreenContext*)
    : torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator(
          [](size_t, int, cudaStream_t) -> void* {
            TORCH_CHECK(
                false,
                "Green Context localization is only supported on CUDA 13.4+!");
          },
          [](void*, size_t, int, cudaStream_t) {
            TORCH_CHECK(
                false,
                "Green Context localization is only supported on CUDA 13.4+!");
          }) {
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
  prop.location.localized.memoryNodeId = green_context->locality_domain_id();
  size_t granularity = 1;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemGetAllocationGranularity_(
    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t domainPaddedSize = (size + (granularity - 1)) & ~(granularity - 1);
  CUmemGenericAllocationHandle handle;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemCreate_(
    &handle, domainPaddedSize, &prop, /*flags*/0));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
    &ptr, domainPaddedSize, granularity, /*addr hint*/0, /*flags*/0));
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
    ptr, domainPaddedSize, /*offset*/0, handle, /*flags*/0));
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemSetAccess_(
    ptr, domainPaddedSize, &desc, /*count [#descriptors]*/1));

  return reinterpret_cast<void*>(ptr);
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return nullptr;
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
  prop.location.localized.memoryNodeId = green_context->locality_domain_id();
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
LocalizedGreenContextMemPool::create_domain_pools(
    std::vector<GreenContext*> const& green_contexts,
    bool alloc_in_order) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  size_t first_idx = green_contexts.size();
  for (size_t i = 0; i < green_contexts.size(); i++) {
    TORCH_CHECK(green_contexts[i]->has_locality_domain(),
      "GreenContext localized MemPool requires localized green context.");
    if (green_contexts[i]->locality_domain_id() == 0) {
      first_idx = i;
      break;
    }
  }
  TORCH_CHECK(
      first_idx < green_contexts.size(),
      "create_domain_pools requires a green context with locality_domain_id 0");
  auto first_pool = std::unique_ptr<LocalizedGreenContextMemPool>(
    new LocalizedGreenContextMemPool(
      green_contexts[first_idx], alloc_in_order, nullptr));
  LocalizedGreenContextMemPool* first_pool_ptr = first_pool.get();
  std::vector<std::unique_ptr<LocalizedGreenContextMemPool>> domain_pools;
  domain_pools.reserve(green_contexts.size());
  for (size_t i = 0; i < green_contexts.size(); i++) {
    if (i == first_idx) {
      domain_pools.push_back(std::move(first_pool));
    } else {
      domain_pools.push_back(std::unique_ptr<LocalizedGreenContextMemPool>(
        new LocalizedGreenContextMemPool(
          green_contexts[i], alloc_in_order, first_pool_ptr)));
    }
  }
  return domain_pools;
#else
  TORCH_CHECK(false, "Green Context localization is only supported on CUDA 13.4+!");
  return {};
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
            if (green_context_->locality_domain_id() == 0) {
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
            first_pool_->domain_events_[green_context_->locality_domain_id()].record(
              c10::cuda::getCurrentCUDAStream());
            green_context_->popContext(/*block_parent_stream=*/false);
            auto last_id =
                first_pool_->green_context()->num_locality_domains() - 1;
            if (green_context_->locality_domain_id() == last_id) {
              for (auto& domain_event : first_pool_->domain_events_) {
                domain_event.block(c10::cuda::getCurrentCUDAStream());
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
      domain_events_(),
      first_pool_(first_pool == nullptr ? this : first_pool) {
#if HAS_CUDA_GREEN_CONTEXT_LOCALIZATION()
  TORCH_CHECK(green_context_, "GreenContext MemPool requires a context.");
  TORCH_CHECK(green_context->has_locality_domain(),
    "GreenContext localized MemPool requires localized green context.");
  if (green_context->num_locality_domains() <= 1) {
    TORCH_WARN_ONCE("GreenContext localized MemPool used with single locality domain.");
  }
  if (first_pool_ == this) {
    domain_events_.resize(green_context->num_locality_domains());
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
} // namespace at::cuda
