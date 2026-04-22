#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <functional>
#include <memory>

namespace at::cuda {

// Keep BC only
using c10::CaptureId_t;
using c10::MempoolId_t;

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the CUDACachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct TORCH_CUDA_CPP_API MemPool {
  // hooks get the device index as an argument and don't return anything
  using HookFnT = std::function<void(int)>;

  MemPool(
      std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator =
          nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false,
      HookFnT on_begin_allocate = nullptr,
      HookFnT on_end_allocate = nullptr);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

  // Optional hooks invoked when entering/leaving the pool in use_mem_pool.
  void call_on_begin_allocate(int device_index);
  void call_on_end_allocate(int device_index);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
  HookFnT on_begin_allocate_;
  HookFnT on_end_allocate_;
};

} // namespace at::cuda
