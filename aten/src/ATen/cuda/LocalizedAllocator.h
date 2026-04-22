#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/NativeAllocatorBackend.h>
#include <cuda.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace at::cuda {

// Native allocator backend that splits allocations into two locality domains.
// Used with NativeAllocatorBackendWrapper so use_native_allocator_backend can switch to it.
class TORCH_CUDA_CPP_API LocalizedAllocator final
    : public c10::cuda::CUDACachingAllocator::NativeAllocatorBackend {
 public:
  static constexpr int MAX_LOCALITY_DOMAINS = 2;
  static constexpr int MIN_ALIGNMENT = 128;
  void* allocate(
      size_t size,
      c10::DeviceIndex device,
      cudaStream_t stream) override;
  void free(void* ptr) override;
  c10::DeleterFnPtr get_deleter() const override;
  std::string name() const override;

 private:
  struct PtrMetadata {
    CUmemGenericAllocationHandle handles[MAX_LOCALITY_DOMAINS];
    LocalizedAllocator* allocator = nullptr;
    size_t domainPaddedSize, domainSize;
    int domainCount;
  };

  void free_(void* ptr, PtrMetadata const& metadata);

  static std::mutex& ptr_mutex_() {
    static std::mutex m;
    return m;
  }
  static std::unordered_map<void*, PtrMetadata>& ptr_to_metadata_() {
    static std::unordered_map<void*, PtrMetadata> map;
    return map;
  }
  static void static_deleter(void* ptr);
};

} // namespace at::cuda
