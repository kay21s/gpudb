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
#include <pthread.h>

#include "./gmm_core.h"
#include "./gmm_type_func.h"


/*********************************** data ***************************************/
gmm_shared gmm_sdata = NULL;
gmm_local gmm_pdata = NULL;
int gmm_id;

sem_t *mutex;
int shm_mc_id = 0;
struct timeval t;
struct timespec wtime = {0, 8388608};

static cudaStream_t mystream = NULL;
static gmm_args targs;


/*********************************** cuda function handlers ***************************************/

static cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*nv_cudaFree)(void *) = NULL;
static cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
static cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t stream) = NULL;
static cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *) = NULL;
static cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t) = NULL;
static cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t) = NULL;
static cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*) = NULL;
static cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
static cudaError_t (*nv_cudaMemset)(void * , int , size_t ) = NULL;	
static cudaError_t (*nv_cudaMemsetAsync)(void * , int , size_t, cudaStream_t) = NULL;	
static cudaError_t (*nv_cudaDeviceSynchronize)(void) = NULL;
static cudaError_t (*nv_cudaLaunch)(void *) = NULL;

/*********************************** gmm functions ***************************************/

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
	init_gmm_shared(gmm_sdata, gpu_mem_free);
	sem_post(mutex);

	return 0;
}


int gmm_reclaim() {
	if (shm_mc_id != 0)
		shmctl(shm_mc_id, IPC_RMID, 0);
	return 0;
}

int gmm_attach() {

	INTERCEPT_CUDA2("cudaMalloc", nv_cudaMalloc);	
	INTERCEPT_CUDA2("cudaFree", nv_cudaFree);	
	INTERCEPT_CUDA2("cudaMemcpy", nv_cudaMemcpy);	
	INTERCEPT_CUDA2("cudaMemcpyAsync", nv_cudaMemcpyAsync);	
	INTERCEPT_CUDA2("cudaStreamCreate", nv_cudaStreamCreate);	
	INTERCEPT_CUDA2("cudaStreamSynchronize", nv_cudaStreamSynchronize);	
	INTERCEPT_CUDA2("cudaSetupArgument", nv_cudaSetupArgument);
	INTERCEPT_CUDA2("cudaConfigureCall", nv_cudaConfigureCall);
	INTERCEPT_CUDA2("cudaMemset", nv_cudaMemset);	
	INTERCEPT_CUDA2("cudaMemsetAsync", nv_cudaMemsetAsync);	
	INTERCEPT_CUDA2("cudaDeviceSynchronize", nv_cudaDeviceSynchronize);	
	INTERCEPT_CUDA2("cudaLaunch", nv_cudaLaunch);	

	//init arg structure for asynchorous functions
	targs = NEW_GMM_ARGS();

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
	
	init_gmm_local(&gmm_pdata);

	sem_wait(mutex);
	gmm_id = new_gmm_id(gmm_sdata);
	S_INC_PNUM(gmm_sdata);
	sem_post(mutex);

	//request for new stream for current process
        if(!mystream)
                nv_cudaStreamCreate(&mystream);

	return 0;
}

int gmm_detach(){
	sem_wait(mutex);
	clean_gmm_shared(gmm_sdata, gmm_id);
	S_DEC_PNUM(gmm_sdata);
	sem_post(mutex);

	sem_close(mutex);
	sem_unlink(GMM_SEM_NAME);
	destroy_gmm_local(gmm_pdata);
	return 0;
}

inline int gmm_getID() {
	return gmm_id;
}

inline void gmm_setFreeMem(size_t size) {
	sem_wait(mutex);
	set_gmm_shared_free(gmm_sdata, size);
	sem_post(mutex);
	return;
}

inline size_t gmm_getFreeMem() {
        return get_gmm_shared_free(gmm_sdata);
}


void gmm_print_sdata() {
	print_gmm_shared(gmm_sdata);
	return;
}


/*****************************Internal Functions************************************/
inline void swap_out() { 
        cudaError_t ret;
	gmm_obj obj = GET_ONE_IN_GPU_OBJ(gmm_pdata);

	GMM_DEBUG(fprintf(stderr, "Swap_out_this_maybe\t") );
	GMM_DEBUG(print_gmm_obj(obj) );

	//malloc space in main memory
	obj->memPtr = (void *) malloc(obj->size);

	//copy obj from GPU mem to main mem (synchronized)
	ret = nv_cudaMemcpy(obj->memPtr, obj->devPtr, obj->size, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		GMM_DEBUG(fprintf(stderr, "Swap_out:Memcpy:Failed \t") );
		GMM_DEBUG(print_gmm_obj(obj) );

		free(obj->memPtr);
		obj->memPtr = NULL;
		return;
	}

	//free GPU memory
	CUDA_SAFE_CALL_NO_SYNC(nv_cudaFree(obj->devPtr));

	//update gmm_pdata
	MV_OBJ_GPU_MAIN(gmm_pdata, obj->key);

	GMM_DEBUG(fprintf(stderr, "Swap_out_success\t") );
	GMM_DEBUG(print_gmm_obj(obj) );

	//update memory usage in gmm_sdata
	sem_wait(mutex);
	S_MV_OBJ_GPU_MAIN(gmm_sdata, gmm_id, obj->size);
	S_INC_MEM_FREE(gmm_sdata, obj->size);
	sem_post(mutex);
	GMM_DEBUG(gmm_print_sdata());

	return;
}

inline void swap_in() { // start within critical region
        cudaError_t ret;

	//leave the critical region, and begin swapping objs into reserved mem
	size_t swappable = S_GET_MEM_SWAPPED(gmm_sdata, gmm_id); 
	size_t reserved = swappable; 
	size_t free = S_GET_MEM_FREE(gmm_sdata);
	if (free <= swappable ) { //can't swap in all objs
		sem_post(mutex);
		GMM_DEBUG(fprintf(stderr, "Swap_in: no enough gpu memory\n") );
		nanosleep(&wtime, NULL);
		return;
	} else {	//can swap in all objs
		S_DEC_MEM_FREE(gmm_sdata, swappable);
		GMM_DEBUG(fprintf(stderr, "Swap_in: %lu Bytes\tFree: %lu\n", swappable, free) );
		sem_post(mutex);
	}	

	void* key = NULL;
	size_t key_size = GMM_KEY_SIZE;
	gmm_obj obj = NULL;
	size_t obj_size = GMM_OBJ_SIZE;

	int hasNext = cfuhash_each_data(gmm_pdata->swapped , &key, &key_size, (void**)(&obj), &obj_size);
	while (hasNext) {
		GMM_DEBUG(fprintf(stderr, "Swap_in_maybe\t") );
		GMM_DEBUG(print_gmm_obj(obj) );
		
		// has enough memory to swap in this obj
		if (obj->memPtr != NULL && reserved >= obj->size ) {
			ret = nv_cudaMalloc(&(obj->devPtr), obj->size);
			if (ret == cudaSuccess) {
				GMM_DEBUG(fprintf(stderr, "Swap_in_success\t") );
				GMM_DEBUG(print_gmm_obj(obj) );

				//copy obj from GPU mem to main mem (synchronized)
				ret = nv_cudaMemcpy(obj->devPtr, obj->memPtr, obj->size, cudaMemcpyHostToDevice);
				//free(obj->memPtr); 
				if (ret != cudaSuccess) {
					GMM_DEBUG(fprintf(stderr, "Swap_in:Memcpy:Failed\t") );
					GMM_DEBUG(print_gmm_obj(obj) );
				}

				GMM_DEBUG(fprintf(stderr, "Swap_in:Memcpy:Completed\t") );
				GMM_DEBUG(print_gmm_obj(obj) );

				//update local info
				MV_OBJ_MAIN_GPU(gmm_pdata, obj->key);
				GMM_DEBUG(print_gmm_obj(obj) );

				
				reserved -= obj->size;	
			} else {//function call failed for reasons other than insufficient memory size
				GMM_DEBUG(fprintf(stderr, "Swap_in:Malloc:Failed \t") );
				GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
				
				//return unused reserved mem
				sem_wait(mutex);
				S_INC_MEM_FREE(gmm_sdata, reserved);
				S_MV_OBJ_MAIN_GPU(gmm_sdata, gmm_id, swappable - reserved);
				sem_post(mutex);
				nanosleep(&wtime, NULL);
				return;
			}

		}
		hasNext = cfuhash_next_data(gmm_pdata->in_gpu_mem, &key, &key_size, (void**)(&obj), &obj_size);
	}

	// update global info
	sem_wait(mutex);
	S_INC_MEM_FREE(gmm_sdata, reserved);
	S_MV_OBJ_MAIN_GPU(gmm_sdata, gmm_id, swappable - reserved);
	sem_post(mutex);

	return;

}


/******************************* Asynchronous Cuda Call Bottom Handler **********************************/
void *cudaLaunch_bh (void * _args) {
	gmm_args args = (gmm_args) _args;
	size_t size = 0; 
	nv_cudaStreamSynchronize(mystream);
	
	GMM_DEBUG(fprintf(stderr, "[Launch_bh]\t") );
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );

	if (args->in_use_num > 0) {
		size = update_local_objs_by_args(gmm_pdata, args);
		sem_wait(mutex);
		S_MV_OBJ_USE_GPU(gmm_sdata, gmm_id, size);
		sem_post(mutex);
	}

	GMM_DEBUG(fprintf(stderr, "[Launch_bh]\tsize: %lu\t", size) );
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
	GMM_DEBUG(print_gmm_local(gmm_pdata) );
	FREE_GMM_ARGS(args);

	return NULL;
}

/******************************* Intercept Cuda Functions **********************************/

inline cudaError_t cudaMalloc(void **devPtr, size_t size) {

	GMM_DEBUG(fprintf(stderr, "gmm_malloc::Entry\tid: %d\tsize: %lu\n", gmm_id, size) );
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
	GMM_DEBUG(gmm_print_sdata());

        cudaError_t ret;
	int min_wait, max_wait;

	while(1) {
		sem_wait(mutex);
		min_wait = get_min_wait(gmm_sdata, gmm_id);	// minimum id > 0 in wait list
		max_wait = get_max_wait(gmm_sdata);		// maximum id whose mu > 0 in the wait list

		GMM_DEBUG(fprintf(stderr, "cudaMalloc::Params\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait) );

		//lowest priority(larger id) + waiting list (>1) + mem usage > 0 = swap out
		if (min_wait < gmm_id && max_wait <= gmm_id &&
			S_GET_MEM_IN_GPU(gmm_sdata, gmm_id) > 0) {
			S_SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);

			GMM_DEBUG(fprintf(stderr, "cudaMalloc::Swap_out\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait) );
			GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );

			swap_out();
			continue;
	
		//else if does not have the highest priority = wait
		} else if (min_wait > 0 && min_wait < gmm_id) {
			S_SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);

			GMM_DEBUG(fprintf(stderr, "cudaMalloc::Wait\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait) );

			// wait & spin
			nanosleep(&wtime, NULL);
			continue;

		//highest priority + swapped = swap in
		} else if (S_GET_MEM_SWAPPED(gmm_sdata, gmm_id) > 0) {
			S_SET_WAIT(gmm_sdata, gmm_id);

			GMM_DEBUG(fprintf(stderr, "cudaMalloc::Swap_in\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait) );

			swap_in(); //where sem_post happens
			continue;

		//highest priority + no swapped obj = just malloc
		} else if ( S_GET_MEM_FREE(gmm_sdata) >= size) {
			S_DEC_MEM_FREE(gmm_sdata, size);
			S_RESET_WAIT(gmm_sdata, gmm_id);	//still have enough memory
			sem_post(mutex);
			ret = nv_cudaMalloc(devPtr, size);
			GMM_DEBUG(fprintf(stderr, "cudaMalloc::malloc\tptr: %p\tdevPtr: %p\tsize: %lu\n", devPtr, *devPtr, size) );
			if (ret == cudaSuccess) {
				if (size <= GMM_IGNORE_SIZE)	{ // do not manage such objs
					sem_wait(mutex);
					S_INC_MEM_CLAIMED(gmm_sdata, gmm_id, size);
					sem_post(mutex);
					NEW_OUTLAW_OBJ(gmm_pdata, devPtr, size);
					break;
				}
				//allocation succeeds, add a new obj in private data
				void* key = NEW_IN_GPU_OBJ(gmm_pdata, (*devPtr), size);
				
				//replace address of the gpu object with home-made key
				*devPtr = key; 

				GMM_DEBUG(gettimeofday(&t, NULL) );
				GMM_DEBUG(fprintf(stderr, "Malloc:success\t[%p]:\t%lf\tKey: %p\tSize: %u\tLeft: %lu\tid: %d\n", devPtr,
					GET_TIMEVAL(t), key, size, S_GET_MEM_FREE(gmm_sdata), gmm_id) );
				GMM_DEBUG(print_gmm_obj2(gmm_pdata, key) );

				//update memory usage of current process
				sem_wait(mutex);
				S_INC_MEM_CLAIMED(gmm_sdata, gmm_id, size);
				S_INC_MEM_IN_GPU(gmm_sdata, gmm_id, size);
				sem_post(mutex);
				break;
			} else {//function call is failed for reasons other than insufficient memory size
				GMM_DEBUG(fprintf(stderr, "cudaMalloc::OUT_OF_REAL_MEM\t") );
				GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );

				sem_wait(mutex);
				S_SET_WAIT(gmm_sdata, gmm_id);	// since the failure may be caused by fragmentation
				S_INC_MEM_FREE(gmm_sdata, size);
				sem_post(mutex);
				nanosleep(&wtime, NULL);
			}
		} else {
			S_SET_WAIT(gmm_sdata, gmm_id);
			sem_post(mutex);
			// wait & spin
			GMM_DEBUG( fprintf(stderr, "cudaMalloc::OUT_OF_MEM\tid: %d\tmin: %d\tmax: %d\n", gmm_id, min_wait, max_wait) );

			nanosleep(&wtime, NULL);
		}
	}

	GMM_DEBUG(fprintf(stderr, "gmm_malloc::Exit\tid: %d\tsize: %lu\n", gmm_id, size) );
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
	GMM_DEBUG(gmm_print_sdata());

	return ret;
}

inline cudaError_t cudaFree(void *_devPtr) {
        cudaError_t ret;

	void * key = _devPtr;
	void * devPtr= _devPtr;
	if (obj_exists(gmm_pdata, key)) {
		devPtr = get_obj_devPtr(gmm_pdata, key);
	}

	GMM_DEBUG(gmm_print_sdata());
	GMM_DEBUG(fprintf(stderr, "cudaFree_ARGS\tkey: %p\tdevPtr: %p\t_devPtr: %p\n", key, devPtr, _devPtr) );
	ret = nv_cudaFree(devPtr);
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
	GMM_DEBUG(print_gmm_obj2(gmm_pdata, key) );
	if (ret == cudaSuccess && key != devPtr) {
		size_t size = get_obj_size(gmm_pdata, key);
		objState state = get_obj_state(gmm_pdata, key);

		sem_wait(mutex);
		update_gmm_shared_remove_obj(gmm_sdata, gmm_id, size, state);
		sem_post(mutex);
		delete_obj(gmm_pdata, key);

		GMM_DEBUG(fprintf(stderr, "cudaFree\tid: %d\tsize: %lu\n", gmm_id, size) );
		GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
		GMM_DEBUG(print_gmm_obj2(gmm_pdata, key) );
	} 
	GMM_DEBUG(gmm_print_sdata());

	return ret;
}


inline cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	GMM_DEBUG(print_ptrs( (void **)arg, size) );
	void* key = NULL;
	cudaError_t ret; 
	if (size == 8)
		key = (*((void **)arg));

	GMM_DEBUG(if(size == 4) fprintf(stderr, "[cudaSetupArgument]\tkey(int): %d\n", *((int*)arg) ));
	GMM_DEBUG(fprintf(stderr, "[cudaSetupArgument]\targ: %p\tsize: %lu\toffset: %lu\tkey: %p\n", arg, size, offset, key) );
	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );

	if (obj_exists(gmm_pdata, key)) {
		void* devPtr = get_obj_devPtr(gmm_pdata, key);
		GMM_DEBUG(fprintf(stderr, "cudaSetupArgument::hasParameter\t*arg: %p\t", devPtr) );
		GMM_DEBUG(print_gmm_obj2(gmm_pdata, key) );
		size_t obj_size = get_obj_size(gmm_pdata, key);
		ret = nv_cudaSetupArgument(&devPtr, size, offset);
		
		//update info
		if (IS_OBJ_IN_GPU(gmm_pdata, key)) {
			sem_wait(mutex);
			S_MV_OBJ_GPU_USE(gmm_sdata, gmm_id, obj_size);
			sem_post(mutex);
			MV_OBJ_GPU_USE(gmm_pdata, key);
			GMM_ARGS_ADD_OBJ(targs, get_obj(gmm_pdata, key));
			//TODO
		}

	} else {
		GMM_DEBUG(fprintf(stderr, "cudaSetupArgument::call\targ: %p\tsize: %lu\toffset: %lu\n", arg, size, offset) );
		ret = nv_cudaSetupArgument(arg, size, offset);
	}

	GMM_DEBUG(print_gmm_shared_info(gmm_sdata, gmm_id) );
	//CUDA_SAFE_CALL_NO_SYNC(nv_cudaDeviceSynchronize() );
	return ret;
}

cudaError_t cudaMemcpy(void *_dst, const void *_src, size_t count, enum cudaMemcpyKind kind) {
	cudaError_t ret;
	void* dst = _dst;
	void* src = (void *)_src;

	//kind = 1	Host2Device
	//kind = 2	Device2Host
	//kind = 3	Device2Device
	if ((kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost) && obj_exists(gmm_pdata, _dst)) {
		dst = get_obj_devPtr(gmm_pdata, _dst);
	}
	if ((kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToHost) && obj_exists(gmm_pdata, (void *)_src)) {
		src = get_obj_devPtr(gmm_pdata, (void *)_src);
	}
	GMM_DEBUG(fprintf(stderr, "[cudaMemcpy]EX\t_dst: %p\t_src: %p\t dst: %p\t src: %p\tkind: %d\tcount: %lu\n", 
		_dst, _src, dst, src, kind, count));
	GMM_DEBUG(gmm_print_sdata());
	GMM_DEBUG(print_gmm_local(gmm_pdata) );
	//ret = nv_cudaMemcpyAsync(dst, src, count, kind, mystream);
	//TODO: memcpy bottom handler
	ret = nv_cudaMemcpy(dst, src, count, kind);
	return ret;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	cudaError_t ret;
	GMM_DEBUG(fprintf(stderr, "[ConfigureCall]\tsharedMem: %lu\tstream: %d\n", sharedMem, stream) );
	GMM_DEBUG2("[ConfigureCall.gridDim]", print_dim3(gridDim));
	GMM_DEBUG2("[ConfigureCall.blockDim]", print_dim3(blockDim));

	ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, mystream);
	//ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	return ret; 
}

cudaError_t cudaMemset(void * _devPtr, int value, size_t count )	 {
	cudaError_t ret;
	void* devPtr = _devPtr;
	if (obj_exists(gmm_pdata, _devPtr)) {
		devPtr = get_obj_devPtr(gmm_pdata, _devPtr);
		GMM_DEBUG(fprintf(stderr, "cudaMemset\tdevPtr: %p\t", devPtr) );
		GMM_DEBUG(print_gmm_obj2(gmm_pdata, _devPtr) );
	}

	//ret = nv_cudaMemsetAsync(devPtr, value, count, mystream);
	ret = nv_cudaMemset(devPtr, value, count);
	return ret;
}

cudaError_t cudaDeviceSynchronize() {
	GMM_DEBUG(fprintf(stderr, "[cudaDeviceSynchronize]\n"));
	return nv_cudaDeviceSynchronize();
}

cudaError_t cudaLaunch (void* entry) {
	GMM_DEBUG(fprintf(stderr, "[cudaLaunch]\n"));
	gmm_args args = NEW_GMM_ARGS();
	CPY_GMM_ARGS(args, targs);
	RESET_GMM_ARGS(targs);
	GMM_DEBUG(print_args(args));

	cudaError_t ret = nv_cudaLaunch(entry);

	pthread_t *pt = (pthread_t *) malloc(sizeof(pthread_t));
	//cudaLaunch_bh (args);
	pthread_create(pt, NULL,  cudaLaunch_bh, args);
	
	return ret;
}

