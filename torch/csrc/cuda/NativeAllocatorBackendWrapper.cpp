#include <ATen/cuda/LocalizedAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/NativeAllocatorBackendWrapper.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda {

NativeAllocatorBackendWrapper::NativeAllocatorBackendWrapper(
    const std::string& path_to_so,
    const std::string& alloc_fn_name,
    const std::string& free_fn_name) {
#if defined(_WIN32) || defined(_WIN64)
  handle_ = LoadLibraryA(path_to_so.c_str());
  TORCH_CHECK(handle_, "Could not load library: ", path_to_so);
  void* alloc_sym =
      (void*)GetProcAddress((HMODULE)handle_, alloc_fn_name.c_str());
  void* free_sym =
      (void*)GetProcAddress((HMODULE)handle_, free_fn_name.c_str());
  TORCH_CHECK(alloc_sym, "Could not find symbol: ", alloc_fn_name);
  TORCH_CHECK(free_sym, "Could not find symbol: ", free_fn_name);
  alloc_fn_ = reinterpret_cast<AllocFnType*>(alloc_sym);
  free_fn_ = reinterpret_cast<FreeFnType*>(free_sym);
#else
  handle_ = dlopen(path_to_so.c_str(), RTLD_LAZY);
  TORCH_CHECK(handle_, "Could not load library: ", path_to_so, ": ", dlerror());
  void* alloc_sym = dlsym(handle_, alloc_fn_name.c_str());
  void* free_sym = dlsym(handle_, free_fn_name.c_str());
  TORCH_CHECK(
      alloc_sym, "Could not find symbol: ", alloc_fn_name, ": ", dlerror());
  TORCH_CHECK(
      free_sym, "Could not find symbol: ", free_fn_name, ": ", dlerror());
  alloc_fn_ = reinterpret_cast<AllocFnType*>(alloc_sym);
  free_fn_ = reinterpret_cast<FreeFnType*>(free_sym);
#endif
}

NativeAllocatorBackendWrapper::~NativeAllocatorBackendWrapper() {
#if defined(_WIN32) || defined(_WIN64)
  if (handle_) {
    FreeLibrary((HMODULE)handle_);
  }
#else
  if (handle_) {
    dlclose(handle_);
  }
#endif
}

void* NativeAllocatorBackendWrapper::allocate(
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
  void* ptr = alloc_fn_(size, static_cast<int>(device), stream);
  if (ptr) {
    std::lock_guard<std::mutex> lock(ptr_mutex_());
    ptr_to_metadata_()[ptr] =
        PtrMetadata{this, size, static_cast<int>(device), stream};
  }
  return ptr;
}

void NativeAllocatorBackendWrapper::free(void* ptr) {
  if (!ptr) {
    return;
  }
  PtrMetadata meta;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(ptr_mutex_());
    auto it = ptr_to_metadata_().find(ptr);
    if (it != ptr_to_metadata_().end()) {
      meta = it->second;
      ptr_to_metadata_().erase(it);
      found = true;
    }
  }
  if (found) {
    free_fn_(ptr, meta.size, meta.device, meta.stream);
  }
}

c10::DeleterFnPtr NativeAllocatorBackendWrapper::get_deleter() const {
  return &static_deleter;
}

std::string NativeAllocatorBackendWrapper::name() const {
  return "custom";
}

void NativeAllocatorBackendWrapper::set_as_native_backend() {
  c10::cuda::CUDACachingAllocator::setNativeAllocatorBackend(this);
}

void NativeAllocatorBackendWrapper::reset_to_caching() {
  c10::cuda::CUDACachingAllocator::resetNativeAllocatorBackendToCaching();
}

void NativeAllocatorBackendWrapper::static_deleter(void* ptr) {
  PtrMetadata meta{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(ptr_mutex_());
    auto it = ptr_to_metadata_().find(ptr);
    if (it != ptr_to_metadata_().end()) {
      meta = it->second;
      ptr_to_metadata_().erase(it);
      found = true;
    }
  }
  if (found && meta.wrapper) {
    meta.wrapper->free_fn_(ptr, meta.size, meta.device, meta.stream);
  }
}

std::mutex& NativeAllocatorBackendWrapper::ptr_mutex_() {
  static std::mutex m;
  return m;
}

std::unordered_map<void*, NativeAllocatorBackendWrapper::PtrMetadata>&
NativeAllocatorBackendWrapper::ptr_to_metadata_() {
  static std::unordered_map<void*, PtrMetadata> map;
  return map;
}

void register_cuda_native_allocator_backend_bindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // Base must be registered before any py::class_ that lists it as a base
  // (otherwise: "referenced unknown base type NativeAllocatorBackend").
  py::class_<
      c10::cuda::CUDACachingAllocator::NativeAllocatorBackend,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::NativeAllocatorBackend>>(
      m, "_NativeAllocatorBackendBase")
      .def(
          "set_as_native_backend",
          [](c10::cuda::CUDACachingAllocator::NativeAllocatorBackend& self) {
            c10::cuda::CUDACachingAllocator::setNativeAllocatorBackend(&self);
          })
      .def(
          "reset_to_caching",
          [](c10::cuda::CUDACachingAllocator::NativeAllocatorBackend&) {
            c10::cuda::CUDACachingAllocator::
                resetNativeAllocatorBackendToCaching();
          });

  py::class_<
      at::cuda::LocalizedAllocator,
      c10::cuda::CUDACachingAllocator::NativeAllocatorBackend,
      std::shared_ptr<at::cuda::LocalizedAllocator>>(
      m, "_CUDALocalizedAllocator")
      .def(py::init(
          []() { return std::make_shared<at::cuda::LocalizedAllocator>(); }))
      .def_readonly_static(
          "min_alignment", &at::cuda::LocalizedAllocator::MIN_ALIGNMENT);

  // .so-based wrapper: loads alloc/free from a shared library.
  py::class_<
      NativeAllocatorBackendWrapper,
      std::shared_ptr<NativeAllocatorBackendWrapper>,
      c10::cuda::CUDACachingAllocator::NativeAllocatorBackend>(
      m, "_NativeAllocatorBackend")
      .def(py::init<
           const std::string&,
           const std::string&,
           const std::string&>());

  m.def("_cuda_getNativeAllocatorBackendName", []() {
    return c10::cuda::CUDACachingAllocator::getNativeAllocatorBackendName();
  });

  m.def("_cuda_resetNativeAllocatorBackendToCaching", []() {
    NativeAllocatorBackendWrapper::reset_to_caching();
  });
}

} // namespace torch::cuda
