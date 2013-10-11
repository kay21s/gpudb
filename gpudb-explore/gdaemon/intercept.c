/*
 * Use nm to list the exact names of function calls in the program before
 * deciding which functions to intercept here.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>                               /* header required for dlsym() */
#include <driver_types.h>
#include <sys/time.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <signal.h>
#include <execinfo.h>

#include "gdaemon.h"

static void show_stackframe() {
  void *trace[32];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  fprintf(stderr, "Printing stack frames:\n");
  for (i=0; i < trace_size; ++i)
        fprintf(stderr, "\t%s\n", messages[i]);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	init_gdaemon();
	addMallocCall(devPtr, size);
	scheduleCudaCall();

	return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr)
{
	init_gdaemon();
	addFreeCall(devPtr);
	scheduleCudaCall();

	return cudaSuccess;
}


cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{

	init_gdaemon();
	addMemcpyCall(dst, src, count, kind);
	scheduleCudaCall();
	
	return cudaSuccess;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	init_gdaemon();
	addConfigureCall(gridDim, blockDim, sharedMem, stream);
	scheduleCudaCall();
	
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	init_gdaemon();
	addSetupCall(arg, size, offset);
	scheduleCudaCall();
	
	return cudaSuccess;
}

cudaError_t cudaLaunch(const char *entry)
{
	init_gdaemon();
	addLaunchCall(entry);
	scheduleCudaCall();
	
	return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void)
{
	init_gdaemon();
	addSyncCall();
	scheduleCudaCall();
	
	return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
	init_gdaemon();
	addMemsetCall(devPtr, value, count);
	scheduleCudaCall();
	
	return cudaSuccess;
}

//CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
//{
//    static CUresult (*nv_cuMemAlloc)(CUdeviceptr *, size_t) = NULL;
//    CUresult res;
//
//    printf("intercepting cuMemAlloc\n");
//
//    if(!nv_cuMemAlloc) {
//        nv_cuMemAlloc = dlsym(RTLD_NEXT, "cuMemAlloc_v2");
//		if(!nv_cuMemAlloc) {
//			perror("failed to find NVIDIA cuMemAlloc\n");
//			return CUDA_ERROR_NOT_FOUND;
//		}
//    }
//
//    res = nv_cuMemAlloc(dptr, bytesize);
//
//    printf("intercepted cuMemAlloc\n");
//
//    return res;
//}

//CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
//{
//    static CUresult (*nv_cuMemAlloc)(CUdeviceptr *, size_t) = NULL;
//    CUresult res;
//
//    printf("intercepting cuMemAlloc\n");
//
//    if(!nv_cuMemAlloc) {
//        nv_cuMemAlloc = dlsym(RTLD_NEXT, "cuMemAlloc");
//		if(!nv_cuMemAlloc) {
//			perror("failed to find NVIDIA cuMemAlloc\n");
//			return CUDA_ERROR_NOT_FOUND;
//		}
//    }
//
//    res = nv_cuMemAlloc(dptr, bytesize);
//
//    printf("intercepted cuMemAlloc\n");
//
//    return res;
//}

//CUresult cuLaunchKernel(CUfunction f,
//						unsigned int gridDimX,
//						unsigned int gridDimY,
//						unsigned int gridDimZ,
//						unsigned int blockDimX,
//						unsigned int blockDimY,
//						unsigned int blockDimZ,
//						unsigned int sharedMemBytes,
//						CUstream hStream,
//						void **kernelParams,
//						void **extra)
//{
//    static CUresult (*nv_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **) = NULL;
//    CUresult res;
//
//    printf("intercepting cuLaunchKernel\n");
//
//    if(!nv_cuLaunchKernel) {
//        nv_cuLaunchKernel = dlsym(RTLD_NEXT, "cuLaunchKernel");
//		if(!nv_cuLaunchKernel) {
//			perror("failed to find NVIDIA cuLaunchKernel\n");
//			return CUDA_ERROR_NOT_FOUND;
//		}
//    }
//
//    res = nv_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
//
//    printf("intercepted cuLaunchKernel\n");
//
//    return res;
//}
