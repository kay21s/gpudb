#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>                               /* header required for dlsym() */
#include <driver_types.h>
#include <sys/time.h>
#include <device_launch_parameters.h>

#include <signal.h>
#include <execinfo.h>
#include <pthread.h>
#include <time.h>

#include <sys/queue.h>
#include "./gd_type.h"

int gd_run = 0;
TAILQ_HEAD(tailhead, entry) gd_head;
pthread_t scheduler;
pthread_mutex_t count_mutex;
struct timespec wtime = {0, 65536};

struct entry {
	cudaCall_sp call;
	TAILQ_ENTRY(entry) entries;
};

void main_destructor( void )
        __attribute__ ((no_instrument_function, destructor));

void main_destructor( void )
{
	if(gd_run == 1) {
		pthread_join(scheduler, NULL);
		gd_run = 0;
	}
}


inline void add_to_queue(cudaCall_sp cl) {
	struct entry *elem;
	elem = malloc(sizeof(struct entry));
	if (elem) {
		elem->call = cl;
	}
	pthread_mutex_lock(&count_mutex);
	TAILQ_INSERT_TAIL(&gd_head, elem, entries);
	pthread_mutex_unlock(&count_mutex);
}



inline void addMemcpyCall(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	argMemcpy_sp args = (argMemcpy_sp) malloc(sizeof(argMemcpy));
	args->dst   = dst;
	args->src   = src;
	args->count = count;
	args->kind  = kind;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDAMEMCPY;
	cl->args = args;

	add_to_queue(cl);
}

inline void addMallocCall(void **devPtr, size_t size) {
	argMalloc_sp args = (argMalloc_sp) malloc(sizeof(argMalloc));
	args->devPtr = devPtr;
	args->size   = size;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDAMALLOC;
	cl->args = args;

	add_to_queue(cl);
}

inline void addFreeCall(void *devPtr) {
	argFree_sp args = (argFree_sp) malloc(sizeof(argFree));
	args->devPtr = devPtr;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDAFREE;
	cl->args = args;

	add_to_queue(cl);
}

inline void addConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	argConfigureCall_sp args = (argConfigureCall_sp) malloc(sizeof(argConfigureCall));
	args->gridDim  = gridDim;
	args->blockDim = blockDim;
	args->sharedMem= sharedMem;
	args->stream   = stream;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDACONFIGURE;
	cl->args = args;

	add_to_queue(cl);
}

inline void addLaunchCall(const char *entry) {
	argLaunch_sp args = (argLaunch_sp) malloc(sizeof(argLaunch));
	args->entry = entry;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDALAUNCH;
	cl->args = args;

	add_to_queue(cl);
}

inline void addSetupCall(const void *arg, size_t size, size_t offset) {
	argSetup_sp args = (argSetup_sp) malloc(sizeof(argSetup));
	args->arg    = arg;
	args->size   = size;
	args->offset = offset;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDASETUP;
	cl->args = args;

	add_to_queue(cl);
}

inline void addMemsetCall(void* devPtr, int value, size_t count){
	argMemset_sp args = (argMemset_sp) malloc(sizeof(argMemset));
	args->devPtr = devPtr;
	args->value  = value;
	args->count  = count;

	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDAMEMSET;
	cl->args = args;

	add_to_queue(cl);
}

inline void addSyncCall(void) {
	cudaCall_sp cl = (cudaCall_sp) malloc(sizeof(cudaCall));
	cl->type  = CUDASYNC;

	add_to_queue(cl);
}

inline cudaError_t gd_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
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

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepted cudaMemcpy( %lx %lx %ld %d ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)dst, (unsigned long)src, count, kind, (int)ret);

        return ret;
}

inline cudaError_t gd_cudaMalloc(void **devPtr, size_t size)
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

        ret = nv_cudaMalloc(devPtr, size);

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepted cudaMalloc( %lx %ld ) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, (unsigned long)(*devPtr), size, (int)ret);

        return ret;
}

inline cudaError_t gd_cudaFree(void *devPtr)
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

inline cudaError_t gd_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
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

inline cudaError_t gd_cudaLaunch(const char *entry)
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

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepted cudaLaunch\n", t.tv_sec + t.tv_usec / 1000000.0);

        return ret;
}

inline cudaError_t gd_cudaSetupArgument(const void *arg, size_t size, size_t offset)
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
        printf(" ) = %d\n", (int)ret);

        return ret;
}

inline cudaError_t gd_cudaThreadSynchronize(void)
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
        printf("[gvm] %lf intercepting cudaThreadSynchronize\n", t.tv_sec + t.tv_usec / 1000000.0);

        ret = nv_cudaThreadSynchronize();

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepted cudaThreadSynchronize\n", t.tv_sec + t.tv_usec / 1000000.0);

        return ret;
}

inline cudaError_t gd_cudaMemset(void* devPtr, int value, size_t count)
{
        static cudaError_t (*nv_cudaMemset)(void*, int, size_t) = NULL;
        cudaError_t ret;
        struct timeval t;

        if(!nv_cudaMemset) {
                nv_cudaMemset= dlsym(RTLD_NEXT, "cudaMemset");
                if(!nv_cudaMemset) {
                        fprintf(stderr, "failed to find symbol cudaMemset: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepting cudaMemset\n", t.tv_sec + t.tv_usec / 1000000.0);

        ret = nv_cudaMemset(devPtr, value, count);

        gettimeofday(&t, NULL);
        printf("[gvm] %lf intercepted cudaMemset( %lx %d %ld) = %d\n", t.tv_sec + t.tv_usec / 1000000.0, 
		(unsigned long)devPtr, value, count, (int)ret);

        return ret;
}

void *gpuScheduleThread(void* dummy) {
	struct entry *elem = NULL;
	cudaCall_sp cl;

	while(1) {
		while (elem = gd_head.tqh_first) {
		cl = elem->call;

			if( cl->type == CUDAMEMCPY ) {
				argMemcpy_sp args = (argMemcpy_sp) cl->args;
				gd_cudaMemcpy(args->dst, args->src, args->count, args->kind);
				free(args);

			} else if (cl->type == CUDAMALLOC) {
				argMalloc_sp args = (argMalloc_sp) cl->args;
				gd_cudaMalloc(args->devPtr, args->size);
				free(args);

			} else if (cl->type == CUDAFREE) {
				argFree_sp args = (argFree_sp) cl->args;
				gd_cudaFree(args->devPtr);
				free(args);
			
			} else if (cl->type == CUDACONFIGURE) {
				argConfigureCall_sp args = (argConfigureCall_sp) cl->args;
				gd_cudaConfigureCall(args->gridDim, args->blockDim, args->sharedMem, args->stream);
				free(args);
			
			} else if (cl->type == CUDALAUNCH) {
				argLaunch_sp args = (argLaunch_sp) cl->args;
				gd_cudaLaunch(args->entry);
				free(args);
			
			} else if (cl->type == CUDASETUP) {
				argSetup_sp args = (argSetup_sp) cl->args;
				gd_cudaSetupArgument(args->arg, args->size, args->offset);
				free(args);
			
			} else if (cl->type == CUDAMEMSET) {
				argMemset_sp args = (argMemset_sp) cl->args;
				gd_cudaMemset(args->devPtr, args->value, args->count);
				free(args);
			
			} else if (cl->type == CUDASYNC) {
				gd_cudaThreadSynchronize();
			}

			pthread_mutex_lock(&count_mutex);
			TAILQ_REMOVE(&gd_head, gd_head.tqh_first, entries);
			pthread_mutex_unlock(&count_mutex);
		}
		nanosleep(&wtime, NULL);
		if(gd_head.tqh_first)
			continue;
		break;
	}
	gd_run = 0;
	return NULL;
}


inline void scheduleCudaCall(void) {
	//gpuScheduleThread(NULL);
}
	
inline void init_gdaemon(void) {
	if(gd_run == 0) {
		TAILQ_INIT(&gd_head);
		pthread_create( &scheduler, NULL, gpuScheduleThread, NULL);
		gd_run = 1;
	}
}
