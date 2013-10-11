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
#include "./gmm_type.h"

#define GET_TIMEVAL(_t) (_t.tv_sec + _t.tv_usec / 1000000.0)

gmm_shared gmm_sdata;
gmm_local gmm_pdata;
int gmm_id;

sem_t *mutex;
int shm_mc_id = 0;

struct timeval t;
struct timespec wtime = {0, 8388608};
static void * obj_ptr = (void*) 17;

static cudaStream_t mystream = NULL;
static cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*nv_cudaFree)(void *) = NULL;
static cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
static cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *) = NULL;
static cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*) = NULL;

inline unsigned int hash(unsigned long int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x % LOCAL_HASH_SIZE;
}

int gmm_init_attach() {

	INTERCEPT_CUDA2("cudaMemGetInfo", nv_cudaMemGetInfo);	

	//create and initialize semaphore
	mutex = sem_open(GMM_SEM_NAME,O_CREAT,0644,1);
	if(mutex == SEM_FAILED) {
		perror("unable to create semaphore");
		sem_unlink(GMM_SEM_NAME);
		exit(-1);
	}

	//create the segment
	if ((shm_mc_id = shmget((key_t)GMM_SHARED, GMM_SHARED_SIZE, IPC_CREAT | 0666)) < 0) {
		perror("shmget");
		exit(1);
	}

	//attach the shared segment
	if ((gmm_sdata = shmat(shm_mc_id, NULL, 0)) == (gmm_shared) -1) {
		perror("shmat");
		exit(1);
	}

	size_t gpu_mem_free = 0;
	size_t gpu_mem_total = 0;
	nv_cudaMemGetInfo(&gpu_mem_free, &gpu_mem_total);
	sem_wait(mutex);
	INIT_GMM_SHARED(gmm_sdata, gpu_mem_free);
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
	if ((shm_mc_id = shmget((key_t)GMM_SHARED, GMM_SHARED_SIZE, 0666)) < 0) {
		perror("shmget");
		exit(1);
	}

	//attach the shared segment
	if ((gmm_sdata = shmat(shm_mc_id, NULL, 0)) == (gmm_shared) -1) {
		perror("shmat");
		exit(1);
	}
	
	gmm_id = NEW_GMM_ID(gmm_sdata);
	MALLOC_LOCAL(gmm_pdata); 

	return 0;
}

int gmm_detach(){
	unsigned long int left;
	
	if (gmm_pdata != NULL) {
		int cur = gmm_pdata->head;
		gmm_obj obj;
		while(cur != -1) {
			obj = &(gmm_pdata->objs[cur]);
			if (obj->memPtr == NULL)
				nv_cudaFree(obj->devPtr);
			cur = gmm_pdata->next[cur];
		}
	}
		
	sem_wait(mutex);
	left = GET_MU(gmm_sdata, gmm_id);	
	INC_MEM_FREE(gmm_sdata, left);
	DEC_MU(gmm_sdata, gmm_id, left);
	RESET_WAIT(gmm_sdata, gmm_id);
	sem_post(mutex);

	sem_close(mutex);
	sem_unlink(GMM_SEM_NAME);
	free(gmm_pdata);
	return 0;
}

inline unsigned long int gmm_getFreeMem() {
	return GET_MEM_FREE(gmm_sdata);
}

inline unsigned long int gmm_getID() {
	return gmm_id;
}

inline void gmm_setFreeMem(unsigned long int size) {
	sem_wait(mutex);
	SET_MEM_FREE(gmm_sdata, size);
	sem_post(mutex);
	return;
}

/*****************************************************************/
inline void print_gmm_sdata() {
	fprintf(stderr, "GPU Memory Available: %ld\tID: %d\n", gmm_getFreeMem(), gmm_getID());
	int i = 0;
	for (i = 0; i < GMM_MU_ARRAY_LEN; i++) {
		if (GET_MU(gmm_sdata, i) > 0)
			fprintf(stderr, "[MU]\tindex: %d\tsize:%lu\n", i, GET_MU(gmm_sdata, i));
	}

	for (i = 0; i < GMM_MU_ARRAY_LEN; i++) {
		if (gmm_sdata->wait[i] > 0)
			fprintf(stderr, "[Wait]\tid: %d\n", gmm_sdata->wait[i]);
	}
}

inline void print_gmm_pdata() {
	int index = gmm_pdata->head;
	gmm_obj obj;
	while (index != -1 ) {
		obj = &(gmm_pdata->objs[index]);
		fprintf(stderr, "index: %d\tptr: %x\tdevPtr: %x\tmemPtr: %x\tsize: %lu\n", 
			index, obj->ptr, obj->devPtr, obj->memPtr, obj->size);
		index = gmm_pdata->next[index];
	}
	return;
}

inline void print_gmm_obj(int index) {
	gmm_obj obj = &(gmm_pdata->objs[index]);
	fprintf(stderr, "index: %d\tptr: %x\tdevPtr: %x\tmemPtr: %x\tsize: %lu\tnext: %d\tprev: %d\n", 
		index, obj->ptr, obj->devPtr, obj->memPtr, obj->size, gmm_pdata->next[index], gmm_pdata->prev[index]);
	return;
}

/*****************************************************************/

inline void* get_one_obj_ptr() {
	obj_ptr++;	
	return obj_ptr;
}

inline size_t set_ignore_objs() { // called after adding a new obj; 
	
	int index = gmm_pdata->head;
	void** head_ptr = (gmm_pdata->objs[index]).ptr;
	gmm_obj obj;
	size_t size = 0;
	index = gmm_pdata->next[index];	// do not set the head
	while (index != -1 ) {
		obj = &(gmm_pdata->objs[index]);
#ifdef GMM_DEBUG
		fprintf(stderr, "Parse_see_if_ignore\t");
		print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
		if (head_ptr == obj->ptr) {	// can no longer swap out this obj
#ifdef GMM_DEBUG
			fprintf(stderr, "Set_ignore\t");
			print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
			size += obj->size;
			DEL_OBJ(gmm_pdata, hash, ((unsigned long int)(obj->devPtr)) );
		}
		index = gmm_pdata->next[index];
	}
	return size; // size of the gpu memory that can not be swapped out
}

inline size_t set_ignore_objs2(void** head_ptr) { 
	
	int index = gmm_pdata->head;
	gmm_obj obj;
	size_t size = 0;
	while (index != -1 ) {
		obj = &(gmm_pdata->objs[index]);
#ifdef GMM_DEBUG
		fprintf(stderr, "Parse_see_if_ignore\t");
		print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
		if (head_ptr == obj->ptr) {	// can no longer swap out this obj
#ifdef GMM_DEBUG
		if (obj->memPtr != NULL)
			fprintf(stderr, "Set_ignore_memPtr_NOTNULL\t");
		else
			fprintf(stderr, "Set_ignore\t");
			print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
			size += obj->size;
			DEL_OBJ(gmm_pdata, hash, ((unsigned long int)(obj->devPtr)) );
		}
		index = gmm_pdata->next[index];
	}
	return size; // size of the gpu memory that can not be swapped out
}

/*****************************************************************/
inline void swap_out() { 
        cudaError_t ret;
	gmm_obj obj;
	int cur = gmm_pdata->head;
	// find one in GPU mem obj
	while(cur != -1) {
		obj = &(gmm_pdata->objs[cur]);
#ifdef GMM_DEBUG
		fprintf(stderr, "Swap_out_this_maybe\tcur :%d\t", cur);
		print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
		if (obj->memPtr == NULL)
			break; //swap out this obj, there MUST be one such obj since mu > 0
		cur = gmm_pdata->next[cur];
	}
	
	//malloc space in main memory
	obj->memPtr = (void *) malloc(obj->size);

	//copy obj from GPU mem to main mem (synchronized)
	ret = nv_cudaMemcpy(obj->memPtr, obj->devPtr, obj->size, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		fprintf(stderr, "Swap_out:Memcpy:Failed \tid: %d\tfrom: %x\tto: %x\tsize: %lu\tmu: %lu\n", gmm_id, obj->memPtr,
			*(obj->ptr), obj->size, GET_MU(gmm_sdata, gmm_id));

		free(obj->memPtr);
		obj->memPtr = NULL;
		return;
	}

	//free GPU memory
	CUDA_SAFE_CALL_NO_SYNC(nv_cudaFree(obj->devPtr));
	gmm_pdata->swapped ++;	

#ifdef GMM_DEBUG
	fprintf(stderr, "Swap_out_DEL\tcur :%d\t", cur);
	print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
	//since current devPtr is very likely to be reused, we adjust curret obj by re-allocating it a useless devPtr
	void* new_ptr = get_one_obj_ptr();
	ADD_OBJ2(gmm_pdata, hash, obj->ptr, ((unsigned long int)new_ptr), obj->memPtr, obj->size);
#ifdef GMM_DEBUG
	fprintf(stderr, "Swap_out_just_ADD\tobj_ptr :%x\t", new_ptr);
	print_gmm_obj(hash((unsigned long int)(new_ptr)));
#endif
	//delete the old obj (obj->ptr is reset to 0)
	DEL_OBJ(gmm_pdata, hash, (unsigned long int)(obj->devPtr) );

#ifdef GMM_DEBUG
	fprintf(stderr, "Swap_out_ADD_a_while\tobj_ptr :%x\t", new_ptr);
	print_gmm_obj(hash((unsigned long int)(new_ptr)));
	fprintf(stderr, "Swap_out_DELELTED\tcur :%d\t", cur);
	print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
	fprintf(stderr, "Swap_out\tid: %d\tfrom: %x\tto: %x\tsize: %lu\tmu: %lu\thead: %d\n", gmm_id, obj->devPtr,
		obj->memPtr, obj->size, GET_MU(gmm_sdata, gmm_id), gmm_pdata->head); 
#endif
	//update memory usage
	sem_wait(mutex);
	DEC_MU(gmm_sdata, gmm_id, obj->size);
	INC_MEM_FREE(gmm_sdata, obj->size);
	sem_post(mutex);

	return;
}

inline void swap_in() { // start within critical region
        cudaError_t ret;
	gmm_obj obj;
	int cur = gmm_pdata->head;
	unsigned long int reserved = 0;
	int swappable = 0;

	while(cur != -1) {
		obj = &(gmm_pdata->objs[cur]);
		
#ifdef GMM_DEBUG
		fprintf(stderr, "Swap_in_maybe_later\t");
		print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
		// has enough memory to swap in this obj
		if (obj->memPtr != NULL && GET_MEM_FREE(gmm_sdata) >= obj->size ) {
			//update memory usage
			INC_MU(gmm_sdata, gmm_id, obj->size);
			DEC_MEM_FREE(gmm_sdata, obj->size);
			reserved += obj->size;
			swappable++;
#ifdef GMM_DEBUG
			fprintf(stderr, "Swap_in_later \tid: %d\tfrom: %x\tto: %x\tsize: %lu\n", gmm_id, obj->memPtr,
				obj->devPtr, obj->size);
#endif
		}
		cur = gmm_pdata->next[cur];
	}
	//leave the critical region, and begin swapping objs into reserved mem
	
	if (reserved == 0 || swappable < GET_SWAPPED(gmm_pdata)) { //can't swap all
		INC_MEM_FREE(gmm_sdata, reserved);
		DEC_MU(gmm_sdata, gmm_id, reserved);
		sem_post(mutex);
		nanosleep(&wtime, NULL);
		return;
	}

	sem_post(mutex);

	cur = gmm_pdata->head;
	while(cur != -1) {
		obj = &(gmm_pdata->objs[cur]);
#ifdef GMM_DEBUG
		fprintf(stderr, "Swap_in_maybe\t");
		print_gmm_obj(hash((unsigned long int)(obj->devPtr)));
#endif
		// has enough memory to swap in this obj
		if (obj->memPtr != NULL && reserved >= obj->size ) {
			ret = nv_cudaMalloc(obj->ptr, obj->size);
			if (ret == cudaSuccess) {
#ifdef GMM_DEBUG
				fprintf(stderr, "Swap_in \tid: %d\tfrom: %x\tto: %x\tsize: %lu\tswapped: %d\n", 
					gmm_id, obj->memPtr, *(obj->ptr), obj->size, GET_SWAPPED(gmm_pdata));
#endif
				//copy obj from GPU mem to main mem (synchronized)
				ret = nv_cudaMemcpy(*(obj->ptr), obj->memPtr, obj->size, cudaMemcpyHostToDevice);
				//free(obj->memPtr); 
				if (ret != cudaSuccess)
					fprintf(stderr, "Swap_in:Memcpy:Failed \tid: %d\tfrom: %x\tto: %x\tsize: %lu\n", gmm_id, obj->memPtr,
						*(obj->ptr), obj->size);
#ifdef GMM_DEBUG
				print_gmm_obj(hash((unsigned long int)(*(obj->ptr) )));
				fprintf(stderr, "Swap_in_copy_completed \tid: %d\tfrom: %x\tto: %x\tsize: %lu\tswapped: %d\n", 
					gmm_id, obj->memPtr, *(obj->ptr), obj->size, GET_SWAPPED(gmm_pdata));
#endif

				//allocation succeeds, add a new obj in private data
				ADD_OBJ(gmm_pdata, hash, obj->ptr, (unsigned long int)(*(obj->ptr)), obj->size);
#ifdef GMM_DEBUG
				print_gmm_obj(hash((unsigned long int)(*(obj->ptr) )));
#endif
				//delete the old obj
				DEL_OBJ(gmm_pdata, hash, (unsigned long int)(obj->devPtr) );
				
				reserved -= obj->size;	
				gmm_pdata->swapped --;	
				cur = gmm_pdata->head;	//just to simplify the logic
			} else {//function call failed for reasons other than insufficient memory size
#ifdef GMM_DEBUG
				fprintf(stderr, "Swap_in:Malloc:Failed \tid: %d\tfrom: %x\tto: %x\tsize: %lu\tswapped: %d\n", 
					gmm_id, obj->memPtr, *(obj->ptr), obj->size, GET_SWAPPED(gmm_pdata));
#endif
				//return unused reserved mem
				sem_wait(mutex);
				INC_MEM_FREE(gmm_sdata, reserved);
				DEC_MU(gmm_sdata, gmm_id, reserved);
				sem_post(mutex);
				nanosleep(&wtime, NULL);
				return;
			}
	
		} 
		cur = gmm_pdata->next[cur];
	}
}

/*****************************************************************/

inline cudaError_t gmm_malloc(void **devPtr, size_t size) {

#ifdef GMM_DEBUG
	fprintf(stderr, "gmm_malloc::Entry\tid: %d\tsize: %lu\tused: %lu\n", gmm_id, size, GET_MU(gmm_sdata, gmm_id));
#endif

	if(!nv_cudaMemcpy) {
                nv_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
                if(!nv_cudaMemcpy) {
                        fprintf(stderr, "failed to find symbol cudaMemcpy: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

	if(!nv_cudaStreamCreate) {
                nv_cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
                if(!nv_cudaStreamCreate) {
                        fprintf(stderr, "failed to find symbol cudaStreamCreate : %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        if(!nv_cudaFree) {
                nv_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
                if(!nv_cudaFree) {
                        fprintf(stderr, "failed to find symbol cudaFree: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        if(!mystream)
                nv_cudaStreamCreate(&mystream);

        if(!nv_cudaMalloc) {
                nv_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
                if(!nv_cudaMalloc) {
                        fprintf(stderr, "failed to find symbol cudaMalloc: %s\n", dlerror());
                        return cudaErrorSharedObjectSymbolNotFound;
                }
        }

        cudaError_t ret;
	int min_wait, max_wait, cur;

	size_t ignore_size = set_ignore_objs2(devPtr);
	sem_wait(mutex);
	DEC_MU(gmm_sdata, gmm_id, ignore_size);
	sem_post(mutex);

	while(1) {
		sem_wait(mutex);
		GET_MIN_WAIT(gmm_sdata, min_wait, gmm_id, cur);	// minimum id > 0 in wait list
		GET_MAX_WAIT(gmm_sdata, max_wait, cur);		// maximum id whose mu > 0 in the wait list

		//lowest priority(larger id) + waiting list (>1) + mem usage > 0 = swap out
		if (min_wait < gmm_id && max_wait <= gmm_id &&
			GET_MU(gmm_sdata, gmm_id) > 0) {
			SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);
#ifdef GMM_DEBUG
			fprintf(stderr, "cudaMalloc::Swap_out\tid: %d\tmin: %d\tmax: %d\tmu: %lu\n", gmm_id, min_wait, max_wait, GET_MU(gmm_sdata, gmm_id));
#endif
			swap_out();
			continue;
	
		//also does not have the highest priority = wait
		} else if (min_wait > 0 && min_wait < gmm_id) {
			SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);
#ifdef GMM_DEBUG
			fprintf(stderr, "cudaMalloc::Wait\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait);
#endif
			// wait & spin
			nanosleep(&wtime, NULL);
			continue;

		//highest priority + swapped = swap in
		} else if (GET_SWAPPED(gmm_pdata) > 0) {
			SET_WAIT(gmm_sdata, gmm_id);
#ifdef GMM_DEBUG
			fprintf(stderr, "cudaMalloc::Swap_in\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait);
#endif
			swap_in(); //where sem_post happens
			continue;

		//highest priority + no swapped obj = just malloc
		} else if ( GET_MEM_FREE(gmm_sdata) >= size) {
			DEC_MEM_FREE(gmm_sdata, size);
			RESET_WAIT(gmm_sdata, gmm_id);	//still have enough memory
			sem_post(mutex);
			ret = nv_cudaMalloc(devPtr, size);
			if (ret == cudaSuccess) {
				if (size <= GMM_IGNORE_SIZE)	{ // do not manage such objs
					break;
				}
				//allocation succeeds, add a new obj in private data
				ADD_OBJ(gmm_pdata, hash, devPtr, (unsigned long int)(*devPtr), size);
#ifdef GMM_DEBUG
				gettimeofday(&t, NULL);
				fprintf(stderr, "Malloc\t[%x]:\t%lf\tptrHash[%lu] = %u\tLeft: %lu\tid: %d\n", *devPtr,
					GET_TIMEVAL(t), hash((unsigned long int)(*devPtr)), size, GET_MEM_FREE(gmm_sdata), gmm_id);
				print_gmm_obj(hash((unsigned long int)(*devPtr)));
#endif
				//update memory usage of current process
				sem_wait(mutex);
				INC_MU(gmm_sdata, gmm_id, size);
				sem_post(mutex);
				break;
			} else {//function call is failed for reasons other than insufficient memory size
#ifdef GMM_DEBUG
				fprintf(stderr, "cudaMalloc::OUT_OF_REAL_MEM\tid: %d\tmin: %d\tmax: %d\tused: %lu\tfree: %lu\n", 
					gmm_id, min_wait, max_wait, GET_MU(gmm_sdata, gmm_id), GET_MEM_FREE(gmm_sdata));
#endif
				sem_wait(mutex);
				SET_WAIT(gmm_sdata, gmm_id);	// since the failure may be caused by fragmentation
				INC_MEM_FREE(gmm_sdata, size);
				sem_post(mutex);
				nanosleep(&wtime, NULL);
			}
		} else {
			SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);
			// wait & spin
#ifdef GMM_DEBUG
			fprintf(stderr, "cudaMalloc::OUT_OF_MEM\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait);
#endif
			nanosleep(&wtime, NULL);
		}
	}

#ifdef GMM_DEBUG
	fprintf(stderr, "gmm_malloc::Exit\tid: %d\tsize: %lu\tused: %lu\n", gmm_id, size, GET_MU(gmm_sdata, gmm_id));
#endif
	return ret;
}

inline cudaError_t gmm_free(void *devPtr) {
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
		size_t size = GET_OBJ_SIZE(gmm_pdata, hash, (unsigned long int)devPtr );

		sem_wait(mutex);
		INC_MEM_FREE(gmm_sdata, size);
		if(OBJ_EXISTS(gmm_pdata, hash, (unsigned long int)devPtr))
			DEC_MU(gmm_sdata, gmm_id, size);
		sem_post(mutex);
		DEL_OBJ(gmm_pdata, hash, (unsigned long int)devPtr );
#ifdef GMM_DEBUG
		fprintf(stderr, "Free\t[%x]:\t%lf\tptrHash[%lu] = %u\tLeft: %lu\tid: %d\n", devPtr, 
			GET_TIMEVAL(t), hash((unsigned long int)(devPtr)), size, GET_MEM_FREE(gmm_sdata), gmm_id);
		print_gmm_obj(hash((unsigned long int)(devPtr)));
#endif
	}

	return ret;
}


CUresult gmm_cuLaunchKernel(CUfunction f,
	unsigned int 	gridDimX,
	unsigned int 	gridDimY,
	unsigned int 	gridDimZ,
	unsigned int 	blockDimX,
	unsigned int 	blockDimY,
	unsigned int 	blockDimZ,
	unsigned int 	sharedMemBytes,
	CUstream 	hStream,
	void ** 	kernelParams,
	void ** 	extra) {

	cudaError_t ret;
	static CUresult (*nv_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, 
		unsigned int, unsigned int, unsigned int, CUstream, void**, void **) = NULL;
	INTERCEPT_CU("cuLaunchKernel", nv_cuLaunchKernel);
	
	int i;
	void* devPtr;
	for (i=0; i<GMM_KERNEL_PARAM_LEN_MAX; i++) {
		devPtr = kernelParams[i];
		if (OBJ_EXISTS(gmm_pdata, hash, (unsigned long int)devPtr)) {
#ifdef GMM_DEBUG
			fprintf(stderr, "cuLaunchKernel::hasParameter\t");
			print_gmm_obj(hash((unsigned long int)(devPtr)));
#endif
		}
	}

	return nv_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

cudaError_t gmm_cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	static cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t) = NULL;
	INTERCEPT_CUDA("cudaSetupArgument", nv_cudaSetupArgument);

	void* devPtr = NULL;
	if (size == 8)
		devPtr = (*((void **)arg));
	if (OBJ_EXISTS(gmm_pdata, hash, (unsigned long int)devPtr)) {
#ifdef GMM_DEBUG
		fprintf(stderr, "cudaSetupArgument::hasParameter\t");
		print_gmm_obj(hash((unsigned long int)(devPtr)));
#endif
		DEL_OBJ(gmm_pdata, hash, ((unsigned long int)(devPtr)) );
		sem_wait(mutex);
		DEC_MU(gmm_sdata, gmm_id, GET_OBJ_SIZE(gmm_pdata, hash, ((unsigned long int)(devPtr))));
		sem_post(mutex);
	}
	
#ifdef GMM_DEBUG
	fprintf(stderr, "cudaSetupArgument::call\targ: %x\tsize: %lu\toffset %lu\n", arg, size, offset);
#endif
	nv_cudaSetupArgument(arg, size, offset);
}
