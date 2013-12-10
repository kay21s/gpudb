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
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <signal.h>
#include <execinfo.h>

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
	static cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaMalloc) {
		nv_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
		//nv_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc_v2");
		if(!nv_cudaMalloc) {
			fprintf(stderr, "failed to find symbol cudaMalloc: %s\n", dlerror());
			show_stackframe();
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaMalloc at %lx\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)devPtr);

	do {
		ret = nv_cudaMalloc(devPtr, size);
	} while (ret != cudaSuccess);

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaMalloc( %lx %ld ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)(*devPtr), size, (int)ret);

	return ret;
}

cudaError_t cudaFree(void *devPtr)
{
	static cudaError_t (*nv_cudaFree)(void *) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaFree) {
		nv_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
		if(!nv_cudaFree) {
			fprintf(stderr, "failed to find symbol cudaFree: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaFree\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaFree(devPtr);

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaFree( %lx ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)devPtr, (int)ret);

	return ret;
}


cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	static cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaMemcpy) {
		nv_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
		if(!nv_cudaMemcpy) {
			fprintf(stderr, "failed to find symbol cudaMemcpy: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaMemcpy\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaMemcpy(dst, src, count, kind);

	cudaThreadSynchronize();

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaMemcpy( %lx %lx %ld %d ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)dst, (unsigned long)src, count, kind, (int)ret);

	return ret;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	static cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaConfigureCall) {
		nv_cudaConfigureCall = dlsym(RTLD_NEXT, "cudaConfigureCall");
		if(!nv_cudaConfigureCall) {
			fprintf(stderr, "failed to find symbol cudaConfigureCall: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaConfigureCall\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaConfigureCall\n", t.tv_sec + t.tv_usec / 1000000.0);

	return ret;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	static cudaError_t (*nv_cudaSetupArgument)(const void *, size_t, size_t) = NULL;
	cudaError_t ret;
	int i;
	struct timeval t;

	if(!nv_cudaSetupArgument) {
		nv_cudaSetupArgument = dlsym(RTLD_NEXT, "cudaSetupArgument");
		if(!nv_cudaSetupArgument) {
			fprintf(stderr, "failed to find symbol cudaSetupArgument: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaSetupArgument\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaSetupArgument(arg, size, offset);

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaSetupArgument( %ld", t.tv_sec + t.tv_usec / 1000000.0, size);
	for(i = 0; i < size; i++) {
		printf(" %x", ((unsigned char *)arg)[i]);
	}
	printf(" ) = %d\n", ret);

	return ret;
}

cudaError_t cudaLaunch(const void *entry)
{
	static cudaError_t (*nv_cudaLaunch)(const char *) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaLaunch) {
		nv_cudaLaunch = dlsym(RTLD_NEXT, "cudaLaunch");
		if(!nv_cudaLaunch) {
			fprintf(stderr, "failed to find symbol cudaLaunch: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepting cudaLaunch\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaLaunch(entry);

	cudaThreadSynchronize();

	gettimeofday(&t, NULL);
	printf("[gvm] %lf intercepted cudaLaunch\n", t.tv_sec + t.tv_usec / 1000000.0);


	return ret;
}

cudaError_t cudaThreadSynchronize(void)
{
	static cudaError_t (*nv_cudaThreadSynchronize)(void) = NULL;
	cudaError_t ret;
	struct timeval t;

	if(!nv_cudaThreadSynchronize) {
		nv_cudaThreadSynchronize = dlsym(RTLD_NEXT, "cudaThreadSynchronize");
		if(!nv_cudaThreadSynchronize) {
			fprintf(stderr, "failed to find symbol cudaThreadSynchronize: %s\n", dlerror());
			return cudaErrorSharedObjectSymbolNotFound;
		}
	}

	gettimeofday(&t, NULL);
	//printf("[gvm] %lf intercepting cudaThreadSynchronize\n", t.tv_sec + t.tv_usec / 1000000.0);

	ret = nv_cudaThreadSynchronize();

	gettimeofday(&t, NULL);
	//printf("[gvm] %lf intercepted cudaThreadSynchronize\n", t.tv_sec + t.tv_usec / 1000000.0);

	return ret;
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
