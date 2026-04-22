#pragma once

#include <ATen/cuda/CUDAGreenContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/MemPool.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include <memory>
#include <vector>

namespace at::cuda {

// Allocator that routes allocations to a specific green context's locality domain.
class TORCH_CUDA_CPP_API LocalizedGreenContextAllocator
    : public torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator {
 public:
  explicit LocalizedGreenContextAllocator(GreenContext* green_context);

 private:
  static void* allocate_on_green_context(
      size_t size,
      int device,
      cudaStream_t stream,
      GreenContext* green_context);
  static void free_on_green_context(
      void* ptr,
      size_t size,
      int device,
      cudaStream_t stream,
      GreenContext* green_context);
};

// Constructed before MemPool, destroyed after -- ensures the allocator
// outlives the pool's destructor (which calls releasePool/emptyCache).
struct LocalizedGreenContextAllocatorHolder {
  std::shared_ptr<LocalizedGreenContextAllocator> allocator;
};

class TORCH_CUDA_CPP_API LocalizedGreenContextMemPool final
    : private LocalizedGreenContextAllocatorHolder,
      public MemPool {
 public:
  GreenContext* green_context() const;

  bool is_alloc_in_order() const;
  void set_alloc_in_order(bool alloc_in_order);

  static std::vector<std::unique_ptr<LocalizedGreenContextMemPool>>
  create_domain_pools(
      std::vector<GreenContext*> const& green_contexts,
      bool alloc_in_order = false);

 private:
  LocalizedGreenContextMemPool(
      GreenContext* green_context,
      bool alloc_in_order,
      LocalizedGreenContextMemPool* first_pool);

  GreenContext* green_context_;
  bool alloc_in_order_;
  at::cuda::CUDAEvent main_event_;
  std::vector<at::cuda::CUDAEvent> domain_events_;
  LocalizedGreenContextMemPool* first_pool_;
};

} // namespace at::cuda
