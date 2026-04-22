#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/NativeAllocatorBackend.h>
#include <torch/csrc/python_headers.h>

#include <mutex>
#include <string>
#include <unordered_map>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace torch::cuda {

// Wrapper that owns state for a custom native allocator backend. Performs
// dlopen to load alloc/free from a .so. Python owns this object; the native
// allocator only holds a non-owning pointer via setNativeAllocatorBackend.
class NativeAllocatorBackendWrapper final
    : public c10::cuda::CUDACachingAllocator::NativeAllocatorBackend {
 public:
  using AllocFnType = void*(size_t, int, cudaStream_t);
  using FreeFnType = void(void*, size_t, int, cudaStream_t);

  struct PtrMetadata {
    NativeAllocatorBackendWrapper* wrapper;
    size_t size;
    int device;
    cudaStream_t stream;
  };

  NativeAllocatorBackendWrapper(
      const std::string& path_to_so,
      const std::string& alloc_fn_name,
      const std::string& free_fn_name);

  ~NativeAllocatorBackendWrapper() override;

  void* allocate(size_t size, c10::DeviceIndex device, cudaStream_t stream)
      override;
  void free(void* ptr) override;
  c10::DeleterFnPtr get_deleter() const override;
  std::string name() const override;

  void set_as_native_backend();
  static void reset_to_caching();

 private:
  static void static_deleter(void* ptr);

  static std::mutex& ptr_mutex_();
  static std::unordered_map<void*, PtrMetadata>& ptr_to_metadata_();

  void* handle_ = nullptr;
  AllocFnType* alloc_fn_ = nullptr;
  FreeFnType* free_fn_ = nullptr;
};

void register_cuda_native_allocator_backend_bindings(PyObject* module);

} // namespace torch::cuda
