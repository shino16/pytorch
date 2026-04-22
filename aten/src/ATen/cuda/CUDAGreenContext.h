#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <cuda.h>

#include <array>
#include <atomic>
#include <memory>
#include <optional>

typedef struct CUgreenCtx_st* CUgreenCtx;

namespace at::cuda {

namespace {
constexpr int kStreamPerGreenContextPool = 32;
}

// Workqueue sharing scope for green contexts.
// Values match the CUDA driver API's CUdevWorkqueueConfigScope enum.
enum class WorkqueueScope : int32_t {
  DeviceCtx = 0,
  Balanced = 1,
};

class LocalizedGreenContextAllocator;

TORCH_CUDA_CPP_API uint32_t get_num_locality_domains(std::optional<uint32_t> device_id);

class TORCH_CUDA_CPP_API GreenContext {
  friend class LocalizedGreenContextAllocator;

 public:
  // Green context creation
  static std::unique_ptr<GreenContext> create(
      std::optional<uint32_t> device_id,
      std::optional<uint32_t> num_sms,
      std::optional<int32_t> workqueue_scope = std::nullopt,
      std::optional<uint32_t> workqueue_concurrency_limit = std::nullopt,
      std::optional<int32_t> locality_domain_id = std::nullopt);

  static uint32_t max_workqueue_concurrency(
      std::optional<uint32_t> device_id = std::nullopt);

  ~GreenContext() noexcept;

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Make this context current
  void setContext(bool block_current_stream = true);
  // Pop this context from the context stack
  void popContext(bool block_parent_stream = true);

  // Create a side stream for this green context
  CUDAStream Stream();

  // Accessors for locality domain and device identity
  int32_t device_id() const;
  int32_t locality_domain_id() const;
  bool has_locality_domain() const;
  int32_t num_locality_domains() const;

 private:
  GreenContext(
      uint32_t device_id,
      std::optional<uint32_t> num_sms,
      std::optional<int32_t> workqueue_scope,
      std::optional<uint32_t> workqueue_concurrency_limit,
      std::optional<int32_t> locality_domain_id);

  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;
  void destroy_resources() noexcept;

  int32_t device_id_ = -1;
  int32_t locality_domain_id_ = -1;
  int32_t num_locality_domains_ = 0;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
  std::array<CUstream, kStreamPerGreenContextPool> green_ctx_streams_{};
  std::atomic<int32_t> curr_stream_idx_ = -1;
};

} // namespace at::cuda
