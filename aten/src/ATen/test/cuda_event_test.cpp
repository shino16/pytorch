#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

TEST(CUDAEventTest, testCUDAExternalEvent) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Create two external CUDA events
  unsigned int flags = cudaEventDefault | cudaEventExternal;
  auto event1 = at::cuda::CUDAEvent(flags);
  auto event2 = at::cuda::CUDAEvent(flags);
  // Ensure external CUDAEvent remain valid and functional after being moved.
  auto start_event = std::move(event1);
  auto end_event = std::move(event2);

  auto stream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(stream);

  auto graph = at::cuda::CUDAGraph();
  graph.capture_begin();
  start_event.record();
  at::cuda::sleep(100000);
  end_event.record();
  graph.capture_end();

  // External events should correctly record timestamps even when used inside
  // CUDA graphs, and elapsed_time() between them should be positive.
  stream.synchronize();
  graph.replay();
  at::cuda::device_synchronize();
  EXPECT_TRUE(start_event.elapsed_time(end_event) > 0);
}

TEST(CUDAEventTest, testCUDAGraphCaptureEpilogueJoinsSecondaryStream) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto capture_stream = at::cuda::getStreamFromPool();
  auto secondary_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard capture_guard(capture_stream);
  at::cuda::CUDAEvent fork_event;
  at::cuda::CUDAEvent join_event;
  at::cuda::CUDAGraph graph;

  graph.capture_begin();
  fork_event.record(capture_stream);
  fork_event.block(secondary_stream);
  {
    at::cuda::CUDAStreamGuard secondary_guard(secondary_stream);
    at::cuda::sleep(1000);
  }
  join_event.record(secondary_stream);

  auto capture_id = c10::cuda::captureIdMayInitCtx(secondary_stream);
  ASSERT_TRUE(capture_id.has_value());
  bool epilogue_ran = false;
  graph.register_capture_epilogue(
      *capture_id,
      [&epilogue_ran, &join_event](const at::cuda::CUDAStream& stream) {
        epilogue_ran = true;
        join_event.block(stream);
      });
  EXPECT_FALSE(epilogue_ran);

  graph.capture_end();
  EXPECT_TRUE(epilogue_ran);
  graph.replay();
  capture_stream.synchronize();
}

namespace {
struct DestructionTracker {
  explicit DestructionTracker(bool* destroyed) : destroyed_(destroyed) {}
  ~DestructionTracker() {
    *destroyed_ = true;
  }
  bool* destroyed_;
};
} // namespace

TEST(CUDAEventTest, testCUDAGraphRetainsCaptureResource) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(stream);
  at::cuda::CUDAGraph graph;
  bool destroyed = false;

  graph.capture_begin();
  at::cuda::sleep(1000);
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  ASSERT_TRUE(capture_id.has_value());
  auto resource = std::make_shared<DestructionTracker>(&destroyed);
  graph.retain_capture_resource(*capture_id, resource);
  resource.reset();
  graph.capture_end();

  EXPECT_FALSE(destroyed);
  graph.reset();
  EXPECT_TRUE(destroyed);
}

TEST(CUDAEventTest, testCUDAGraphEpilogueFailureEndsCapture) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(stream);
  at::cuda::CUDAGraph graph;
  bool later_epilogue_ran = false;
  bool resource_destroyed = false;

  graph.capture_begin();
  at::cuda::sleep(1000);
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  ASSERT_TRUE(capture_id.has_value());
  auto resource = std::make_shared<DestructionTracker>(&resource_destroyed);
  graph.retain_capture_resource(*capture_id, resource);
  resource.reset();
  graph.register_capture_epilogue(*capture_id, [](const at::cuda::CUDAStream&) {
    TORCH_CHECK(false, "expected capture epilogue failure");
  });
  graph.register_capture_epilogue(
      *capture_id, [&later_epilogue_ran](const at::cuda::CUDAStream&) {
        later_epilogue_ran = true;
      });

  EXPECT_THROW(graph.capture_end(), c10::Error);
  EXPECT_TRUE(later_epilogue_ran);
  EXPECT_TRUE(resource_destroyed);
  EXPECT_EQ(
      c10::cuda::currentStreamCaptureStatusMayInitCtx(),
      c10::cuda::CaptureStatus::None);
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040
TEST(CUDAEventTest, testCUDAGraphConditionalCaptureEpilogue) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(stream);
  auto predicate =
      at::zeros({}, at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
  at::cuda::CUDAGraph graph;
  bool epilogue_ran = false;

  graph.capture_begin();
  graph.begin_capture_to_if_node(predicate);
  at::cuda::sleep(1000);
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  ASSERT_TRUE(capture_id.has_value());
  graph.register_capture_epilogue(
      *capture_id,
      [&epilogue_ran](const at::cuda::CUDAStream&) { epilogue_ran = true; });
  graph.end_capture_to_conditional_node();
  graph.capture_end();

  EXPECT_TRUE(epilogue_ran);
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  return RUN_ALL_TESTS();
}
