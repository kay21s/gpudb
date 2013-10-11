#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <semaphore.h>

#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>			/* header required for dlsym() */
#include <driver_types.h>
#include <sys/time.h>
#include <time.h>
#include <fcntl.h>			/* For O_* constants */
#include <errno.h>
#include <sys/stat.h>			/* For mode constants */
#include <sys/mman.h>

#include "./gmm_meta.h"

#define PTR_HASH_SIZE 262144
#define GET_TIMEVAL(_t) (_t.tv_sec + _t.tv_usec / 1000000.0)

unsigned long int *gmm_mc;
sem_t *mutex;
int shm_mc_id = 0;

size_t gpu_mem_free;
size_t gpu_mem_total;
struct timeval t;

struct timespec wtime = {0, 65536};
size_t ptrHash[PTR_HASH_SIZE];	//size

inline unsigned int hash(unsigned long int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x % PTR_HASH_SIZE;
}

int gmm_init_attach(unsigned long int mem_size) {
	//static cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*) = NULL;

        //if(!nv_cudaMemGetInfo) {
        //        nv_cudaMemGetInfo= dlsym(RTLD_NEXT, "cudaMemGetInfo");
        //        if(!nv_cudaMemGetInfo) {
        //                fprintf(stderr, "failed to find symbol cudaMemGetInfo: %s\n", dlerror());
        //                return cudaErrorSharedObjectSymbolNotFound;
        //        }
        //}

	//create and initialize semaphore
	mutex = sem_open(GMM_SEM_NAME,O_CREAT,0644,1);
	if(mutex == SEM_FAILED) {
		perror("unable to create semaphore");
		sem_unlink(GMM_SEM_NAME);
		exit(-1);
	}

	//create the segment
	if ((shm_mc_id = shmget((key_t)GMM_MEM_COUNTER, GMM_MEM_COUNTER_SIZE, IPC_CREAT | 0666)) < 0) {
		perror("shmget");
		exit(1);
	}

	//attach the shared segment
	if ((gmm_mc = shmat(shm_mc_id, NULL, 0)) == (unsigned long int *) -1) {
		perror("shmat");
		exit(1);
	}

	//nv_cudaMemGetInfo(&gpu_mem_free, &gpu_mem_total);
	sem_wait(mutex);
	*gmm_mc = mem_size; 
	sem_post(mutex);

	return 0;
}


int gmm_reclaim() {
	if (shm_mc_id != 0)
		shmctl(shm_mc_id, IPC_RMID, 0);
	return 0;
}

int gmm_attach() {

	//create and initialize semaphore
	mutex = sem_open(GMM_SEM_NAME,O_CREAT,0644,1);
	if(mutex == SEM_FAILED) {
		perror("unable to create semaphore");
		sem_unlink(GMM_SEM_NAME);
		exit(-1);
	}

	//get the segment id
	if ((shm_mc_id = shmget((key_t)GMM_MEM_COUNTER, GMM_MEM_COUNTER_SIZE, 0666)) < 0) {
		perror("shmget");
		exit(1);
	}

	//attach the shared segment
	if ((gmm_mc = shmat(shm_mc_id, NULL, 0)) == (unsigned long int *) -1) {
		perror("shmat");
		exit(1);
	}

	return 0;
}

int gmm_detach(){

	sem_close(mutex);
	sem_unlink(GMM_SEM_NAME);
	return 0;
}

unsigned long int gmm_getMC() {
	return *gmm_mc;
}

inline void gmm_setMC(unsigned long int size) {
	sem_wait(mutex);
	*gmm_mc = size;
	sem_post(mutex);
	return;
}

inline cudaError_t gmm_malloc(void **devPtr, size_t size) {
	static cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
        cudaError_t ret;

        if(!nv_cudaMalloc) {
                nv_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
                if(!nv_cudaMalloc) {
                        fprintf(stderr, "failed to find symbol cudaMalloc: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

	while(1) {
		sem_wait(mutex);
		if ((*gmm_mc) >= size) {
			*gmm_mc = (*gmm_mc) - size;
			sem_post(mutex);
			ret = nv_cudaMalloc(devPtr, size);
			if (ret == cudaSuccess) {
				ptrHash[hash((unsigned long int)(*devPtr))] = size;
#ifdef GMM_DEBUG
				gettimeofday(&t, NULL);
				fprintf(stdout, "Malloc\t[%x]:\t%lf\tptrHash[%lu] = %u\tLeft: %lu\n", *devPtr,
					GET_TIMEVAL(t), hash((unsigned long int)(*devPtr)), size, *gmm_mc);
#endif
				break;
			} else { // function call is failed for reasons other than insufficient memory size
				sem_wait(mutex);
				*gmm_mc = (*gmm_mc) + size;
				sem_post(mutex);
				nanosleep(&wtime, NULL);
			}
		} else {
			sem_post(mutex);
			// wait & spin
			nanosleep(&wtime, NULL);
		}
	}

	return ret;
}

inline cudaError_t gmm_free(void *devPtr) {
        static cudaError_t (*nv_cudaFree)(void *) = NULL;
        cudaError_t ret;

        if(!nv_cudaFree) {
                nv_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
                if(!nv_cudaFree) {
                        fprintf(stderr, "failed to find symbol cudaFree: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

	ret = nv_cudaFree(devPtr);
	if (ret == cudaSuccess) {
		size_t size = ptrHash[hash((unsigned long int)devPtr)];

		sem_wait(mutex);
		*gmm_mc = (*gmm_mc) + size;
		sem_post(mutex);
#ifdef GMM_DEBUG
		fprintf(stdout, "Free\t[%x]:\t%lf\tptrHash[%lu] = %u\tLeft: %lu\n", devPtr, 
			GET_TIMEVAL(t), hash((unsigned long int)(devPtr)), size, *gmm_mc);
#endif
	}

	return ret;
}
