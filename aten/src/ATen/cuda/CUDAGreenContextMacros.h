#pragma once

// Feature-detection macros for CUDA green contexts:
//   HAS_CUDA_GREEN_CONTEXT()              base API           (CUDA 12.8+)
//   HAS_CUDA_WORKQUEUE_SUPPORT()          workqueue resource (CUDA 13.1+)
//   HAS_CUDA_GREEN_CONTEXT_LOCALIZATION() locality domains   (CUDA 13.4+)
//
// All require PYTORCH_C10_DRIVER_API_SUPPORTED for the translation unit (set
// via per-source COMPILE_FLAGS in caffe2/CMakeLists.txt).

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && !defined(USE_ROCM) && \
    defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#define HAS_CUDA_GREEN_CONTEXT() 1
#else
#define HAS_CUDA_GREEN_CONTEXT() 0
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010 && HAS_CUDA_GREEN_CONTEXT()
#define HAS_CUDA_WORKQUEUE_SUPPORT() 1
#else
#define HAS_CUDA_WORKQUEUE_SUPPORT() 0
#endif

#if HAS_CUDA_GREEN_CONTEXT() == 1 && CUDA_VERSION >= 13040
#define HAS_CUDA_GREEN_CONTEXT_LOCALIZATION() 1
#else
#define HAS_CUDA_GREEN_CONTEXT_LOCALIZATION() 0
#endif
