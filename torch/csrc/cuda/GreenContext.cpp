#include <ATen/cuda/CUDAGreenContext.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def(
      "_get_num_locality_domains",
      &at::cuda::get_num_locality_domains,
      py::arg("device_id") = py::none());

  py::enum_<at::cuda::WorkqueueScope>(m, "_WorkqueueScope")
      .value("device_ctx", at::cuda::WorkqueueScope::DeviceCtx)
      .value("balanced", at::cuda::WorkqueueScope::Balanced);

  py::class_<at::cuda::GreenContext>(m, "_CUDAGreenContext")
      .def_static(
          "create",
          [](std::optional<uint32_t> device_id,
             std::optional<uint32_t> num_sms,
             std::optional<std::string> workqueue_scope,
             std::optional<uint32_t> workqueue_concurrency_limit,
             std::optional<int32_t> locality_domain_id) {
            std::optional<int32_t> scope;
            if (workqueue_scope.has_value()) {
              const auto& s = *workqueue_scope;
              if (s == "device_ctx") {
                scope =
                    static_cast<int32_t>(at::cuda::WorkqueueScope::DeviceCtx);
              } else if (s == "balanced") {
                scope =
                    static_cast<int32_t>(at::cuda::WorkqueueScope::Balanced);
              } else {
                throw std::invalid_argument(
                    "workqueue_scope must be 'device_ctx' or 'balanced', got '" +
                    s + "'");
              }
            }
            return at::cuda::GreenContext::create(
                device_id,
                num_sms,
                scope,
                workqueue_concurrency_limit,
                locality_domain_id);
          },
          py::kw_only(),
          py::arg("device_id") = py::none(),
          py::arg("num_sms") = py::none(),
          py::arg("workqueue_scope") = py::none(),
          py::arg("workqueue_concurrency_limit") = py::none(),
          py::arg("locality_domain_id") = py::none())
      .def_static(
          "max_workqueue_concurrency",
          &at::cuda::GreenContext::max_workqueue_concurrency,
          py::arg("device_id") = py::none())
      .def_property_readonly("device_id", &::at::cuda::GreenContext::device_id)
      .def_property_readonly(
          "locality_domain_id", &::at::cuda::GreenContext::locality_domain_id)
      .def_property_readonly(
          "num_locality_domains",
          &::at::cuda::GreenContext::num_locality_domains)
      .def_property_readonly(
          "has_locality_domain", &::at::cuda::GreenContext::has_locality_domain)
      .def(
          "set_context",
          &::at::cuda::GreenContext::setContext,
          py::arg("block_current_stream") = true)
      .def(
          "pop_context",
          &::at::cuda::GreenContext::popContext,
          py::arg("block_parent_stream") = true)
      .def("Stream", [](at::cuda::GreenContext& self) {
        auto s = self.Stream();
        cudaStream_t raw = s.stream();
        auto ptr_val = reinterpret_cast<uintptr_t>(raw);

        py::object torch_cuda = py::module::import("torch.cuda");
        py::object ExternalStream = torch_cuda.attr("ExternalStream");

        return ExternalStream(ptr_val, py::int_(s.device_index()));
      });
}
