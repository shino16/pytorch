#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include <mutex>
#include <unordered_map>
#include <memory>
#include <vector>

// Forward declare green context as opaque ptr
typedef struct CUgreenCtx_st* CUgreenCtx;

namespace at::cuda {

namespace {
  constexpr int kStreamPerGreenContextPool = 32;
}

TORCH_CUDA_CPP_API uint32_t get_num_memory_nodes(uint32_t device_id);

class TORCH_CUDA_CPP_API LocalizedGreenContextAllocator;

class TORCH_CUDA_CPP_API GreenContext {
  friend class LocalizedGreenContextAllocator;

 public:
  // Green context creation
  static std::unique_ptr<GreenContext> create(
      uint32_t num_sms,
      std::optional<uint32_t> device_id);
  static std::vector<std::unique_ptr<GreenContext>> create_localized(
    std::optional<uint32_t> device_id);
  ~GreenContext() noexcept;

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Make this context current
  void setContext(bool block_current_stream = true);

  // Pop this context from the context stack
  void popContext(bool block_parent_stream = true);

  CUDAStream Stream();

  // Accessors for memory locality and device identity
  int32_t device_id() const;
  int32_t memory_node_id() const;
  bool has_memory_node() const;
  int32_t num_memory_nodes() const;

 private:
  GreenContext(uint32_t device_id, uint32_t num_sms);
  GreenContext(int32_t device_id, int32_t memory_node_id, int32_t num_memory_nodes);
  // Implement move operations
  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;
  void destroy_resources() noexcept;

  int32_t device_id_ = -1;
  int32_t memory_node_id_ = -1;
  int32_t num_memory_nodes_ = 0;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
  std::array<CUstream, kStreamPerGreenContextPool> green_ctx_streams_;
  std::atomic<int32_t> curr_stream_idx_ = -1;
};

// Allocator that routes allocations to a specific green context's memory node.
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
  create_node_pools(
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
  std::vector<at::cuda::CUDAEvent> node_events_;
  LocalizedGreenContextMemPool* first_pool_;
};

// Native allocator backend for localized (e.g. uGPU memory-node) allocation.
// Used with NativeAllocatorBackendWrapper so use_native_allocator_backend can switch to it.
class TORCH_CUDA_CPP_API LocalizedAllocator final
    : public c10::cuda::CUDACachingAllocator::NativeAllocatorBackend {
 public:
  static constexpr int MAX_MEMORY_NODES = 2;
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
    CUmemGenericAllocationHandle handles[MAX_MEMORY_NODES];
    LocalizedAllocator* allocator = nullptr;
    size_t nodePaddedSize, nodeSize;
    int memoryNodeCount;
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
