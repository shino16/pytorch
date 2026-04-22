#include <ATen/cuda/LocalizedAllocator.h>
#include <ATen/cuda/CUDAGreenContextMacros.h>

#if HAS_CUDA_GREEN_CONTEXT()
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#endif

namespace at::cuda {

// ---------------------------------------------------------------------------
// LocalizedAllocator (NativeAllocatorBackend for localized allocation)
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
    "Currently the localized allocator only supports 2 locality domains.");
  TORCH_CHECK(size % count == 0, "Size ", size,
    " must be a multiple of the number of locality domains: ", count);

  TORCH_INTERNAL_ASSERT(count > 0 && count <= MAX_LOCALITY_DOMAINS,
    "Invalid number of locality domains: ", count);

  // Create the memory allocation on the two locality domains.
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
      "granularity must be the same for all locality domains");
  }

  // Each domain-allocation size must be a multiple of granularity (driver API
  // requirement). So round each domain's memory up to granularity.
  // in order to guarantee support for any C++ data type, we first ensure that
  // the domain size is a multiple of 16 bytes.
  TORCH_INTERNAL_ASSERT(granularity % MIN_ALIGNMENT == 0,
    "granularity must be a multiple of MIN_ALIGN");
  size_t domainSize = size / count;
  domainSize = (domainSize + MIN_ALIGNMENT - 1) & ~(MIN_ALIGNMENT - 1);
  size_t domainPaddedSize = (domainSize + (granularity - 1)) & ~(granularity - 1);
  size_t paddedSize = domainPaddedSize * count;
  TORCH_INTERNAL_ASSERT(domainPaddedSize >= domainSize,
    "domainPaddedSize must be greater or equal to domainSize");

  PtrMetadata metadata{};
  metadata.allocator = this;
  metadata.domainPaddedSize = domainPaddedSize;
  metadata.domainSize = domainSize;
  metadata.domainCount = count;

  for (int i = 0; i < count; i++) {
    prop.location.localized.memoryNodeId = i;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemCreate_(
      &metadata.handles[i],
      domainPaddedSize,
      &prop,
      /*flags*/0));
  }

  CUdeviceptr devPtr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
    &devPtr, paddedSize, granularity, /*addr hint*/0, /*flags*/0));
  // cuMemMap requires ptr and size to be multiples of allocation granularity.
  // Thus, we will center any allocation around the granularity boundary.
  // Note that this does not work in case we have more than 2 locality domains.
  for (int i = 0; i < count; i++) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
      devPtr + i * domainPaddedSize, domainPaddedSize, /*offset*/0, metadata.handles[i], /*flags*/0));
  }

  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemSetAccess_(
    devPtr, paddedSize, &desc, /*count [#descriptors]*/1));

  // here we center the output pointer around the granularity boundary.
  auto* ptr = reinterpret_cast<void*>(devPtr + (domainPaddedSize - domainSize));
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
    (metadata.domainPaddedSize - metadata.domainSize);
  for (int i = 0; i < metadata.domainCount; i++) {
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemUnmap_(
      devPtr + i * metadata.domainPaddedSize, metadata.domainPaddedSize));
  }
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemAddressFree_(
    devPtr, metadata.domainPaddedSize * metadata.domainCount));
  for (int i = 0; i < metadata.domainCount; i++) {
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
