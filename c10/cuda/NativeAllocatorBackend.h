#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

namespace c10::cuda::CUDACachingAllocator {

// Abstract backend for the native allocator's low-level allocate/free.
// Implementations can be provided by C++ or the python process.
struct C10_CUDA_API NativeAllocatorBackend {
  virtual void* allocate(
      size_t size,
      c10::DeviceIndex device,
      cudaStream_t stream) = 0;
  virtual void free(void* ptr) = 0;
  virtual c10::DeleterFnPtr get_deleter() const = 0;
  virtual std::string name() const = 0;
  virtual ~NativeAllocatorBackend() = default;
};

// Native allocator backend get/set. Only apply when the current allocator is
// the native one (name() == "native"). When PYTORCH_NO_CUDA_MEMORY_CACHING is
// set, set_backend and reset are no-ops. The backend pointer is not owned;
// the caller (e.g. Python) must keep the backend alive.
C10_CUDA_API std::string getNativeAllocatorBackendName();
C10_CUDA_API void setNativeAllocatorBackend(NativeAllocatorBackend* backend);
C10_CUDA_API void resetNativeAllocatorBackendToCaching();

} // namespace c10::cuda::CUDACachingAllocator
