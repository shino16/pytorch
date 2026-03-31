#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGreenContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

// Cargo culted partially from csrc/cuda/Stream.cpp

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_get_num_memory_nodes", &at::cuda::get_num_memory_nodes);

  py::class_<
      at::cuda::LocalizedAllocator,
      c10::cuda::CUDACachingAllocator::NativeAllocatorBackend,
      std::shared_ptr<at::cuda::LocalizedAllocator>>(
      m, "_CUDALocalizedAllocator")
      .def(py::init(
          []() { return std::make_shared<at::cuda::LocalizedAllocator>(); }))
      .def_readonly_static(
          "min_alignment", &at::cuda::LocalizedAllocator::MIN_ALIGNMENT);

  py::class_<at::cuda::GreenContext>(m, "_CUDAGreenContext")
      .def_static("create", &::at::cuda::GreenContext::create)
      .def_static(
          "create_localized", &::at::cuda::GreenContext::create_localized)
      .def_property_readonly("device_id", &::at::cuda::GreenContext::device_id)
      .def_property_readonly(
          "memory_node_id", &::at::cuda::GreenContext::memory_node_id)
      .def_property_readonly(
          "num_memory_nodes", &::at::cuda::GreenContext::num_memory_nodes)
      .def_property_readonly(
          "has_memory_node", &::at::cuda::GreenContext::has_memory_node)
      .def(
          "set_context",
          &::at::cuda::GreenContext::setContext,
          py::arg("block_current_stream"))
      .def(
          "pop_context",
          &::at::cuda::GreenContext::popContext,
          py::arg("block_parent_stream"))
      .def("Stream", [](at::cuda::GreenContext& self) {
        auto s = self.Stream();
        cudaStream_t raw = s.stream();
        auto ptr_val = reinterpret_cast<uintptr_t>(raw);

        py::object torch_cuda = py::module::import("torch.cuda");
        py::object ExternalStream = torch_cuda.attr("ExternalStream");

        return ExternalStream(ptr_val, py::int_(s.device_index()));
      });

  using Pool = at::cuda::LocalizedGreenContextMemPool;
  py::class_<Pool, std::shared_ptr<Pool>>(
      m, "_CUDALocalizedGreenContextMemPool")
      .def_static(
          "create_node_pools",
          [](py::list green_context_list, bool alloc_in_order) {
            torch::utils::device_lazy_init(at::kCUDA);
            std::vector<at::cuda::GreenContext*> ptrs;
            for (auto& obj : green_context_list) {
              ptrs.push_back(obj.cast<at::cuda::GreenContext*>());
            }
            auto pools = Pool::create_node_pools(ptrs, alloc_in_order);
            // Pools hold raw GreenContext*; the contexts are owned by
            // green_context_list. Each shared_ptr's custom deleter captures
            // keepalive so the list outlives every pool (avoids use-after-free
            // if the user keeps e.g. only pools[0] and drops the list).
            py::object keepalive = green_context_list;
            py::list result;
            for (auto& p : pools) {
              result.append(py::cast(std::shared_ptr<Pool>(
                  p.release(), [keepalive](Pool* ptr) { delete ptr; })));
            }
            return result;
          },
          py::arg("green_contexts"),
          py::arg("alloc_in_order") = false)
      .def_property_readonly("id", &Pool::id)
      .def_property(
          "alloc_in_order", &Pool::is_alloc_in_order, &Pool::set_alloc_in_order)
      .def("use_count", &Pool::use_count)
      .def("call_on_begin_allocate", &Pool::call_on_begin_allocate)
      .def("call_on_end_allocate", &Pool::call_on_end_allocate)
      .def_property_readonly(
          "green_context",
          &Pool::green_context,
          py::return_value_policy::reference);
}
