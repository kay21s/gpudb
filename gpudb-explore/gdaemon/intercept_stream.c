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

static cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *) = NULL;	
static cudaStream_t mystream = NULL;

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



cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
        static cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t stream) = NULL;
        cudaError_t ret;
        struct timeval t;
        
	if(!nv_cudaStreamCreate) {
                nv_cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
                if(!nv_cudaStreamCreate) {
                        fprintf(stderr, "failed to find symbol cudaStreamCreate : %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
	}	
	
	if(!mystream)
		ret = nv_cudaStreamCreate(&mystream);

        if(!nv_cudaMemcpy) {
                nv_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
                if(!nv_cudaMemcpy) {
                        fprintf(stderr, "failed to find symbol cudaMemcpy: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        //gettimeofday(&t, NULL);
	//printf("[gvm] %lf intercepting cudaMemcpy\n", t.tv_sec + t.tv_usec / 1000000.0);

        ret = nv_cudaMemcpy(dst, src, count, kind, mystream);

        //gettimeofday(&t, NULL);
        //printf("[gvm] %lf intercepted cudaMemcpy( %lx %lx %ld %d ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)dst, (unsigned long)src, count, kind, (int)ret);

        return ret;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
        static cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
        cudaError_t ret;
        struct timeval t;

        if(!nv_cudaStreamCreate) {
                nv_cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
                if(!nv_cudaStreamCreate) {
                        fprintf(stderr, "failed to find symbol cudaStreamCreate : %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
	}	
	
	if(!mystream)
		ret = nv_cudaStreamCreate(&mystream);

        if(!nv_cudaConfigureCall) {
                nv_cudaConfigureCall = dlsym(RTLD_NEXT, "cudaConfigureCall");
                if(!nv_cudaConfigureCall) {
                        fprintf(stderr, "failed to find symbol cudaConfigureCall: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        //gettimeofday(&t, NULL);
        //printf("[gvm] %lf intercepting cudaConfigureCall\n", t.tv_sec + t.tv_usec / 1000000.0);

        ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, mystream);

        //gettimeofday(&t, NULL);
        //printf("[gvm] %lf intercepted cudaConfigureCall ( %d ) \n", t.tv_sec + t.tv_usec / 1000000.0, (int) mystream);

        return ret;
}
