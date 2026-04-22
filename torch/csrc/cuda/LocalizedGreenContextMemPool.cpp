#include <ATen/cuda/CUDAGreenContext.h>
#include <ATen/cuda/LocalizedGreenContextMemPool.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

void THCPLocalizedGreenContextMemPool_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  using Pool = at::cuda::LocalizedGreenContextMemPool;
  py::class_<Pool, std::shared_ptr<Pool>>(
      m, "_CUDALocalizedGreenContextMemPool")
      .def_static(
          "create_domain_pools",
          [](py::list green_context_list, bool alloc_in_order) {
            torch::utils::device_lazy_init(at::kCUDA);
            std::vector<at::cuda::GreenContext*> ptrs;
            for (auto& obj : green_context_list) {
              ptrs.push_back(obj.cast<at::cuda::GreenContext*>());
            }
            auto pools = Pool::create_domain_pools(ptrs, alloc_in_order);
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
