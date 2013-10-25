// Include this file in user program to access GMM-specific features.
#ifndef _GMM_EXT_H_
#define _GMM_EXT_H_

#include <cuda_runtime_api.h>

// Read/write hints
#define HINT_READ	1
#define HINT_WRITE	2
#define HINT_DEFAULT	(HINT_READ | HINT_WRITE)

#define GMM_STUB_DECLARE(func_)	cudaError_t func_ { return cudaSuccess;}

#ifdef __cplusplus
extern "C" {
#endif

// The GMM extensions to CUDA runtime interfaces. Interface
// implementations reside in interfaces.c.
cudaError_t cudaSetKernelPrio(int prio);
cudaError_t cudaReference(int which_arg, int flags);

#ifdef __cplusplus
}
#endif

#endif
