#ifndef _GMM_TYPE_H_
#define _GMM_TYPE_H_

#include <cfuhash.h>

// by default, shmget initialize all values to 0
#define GMM_MU_ARRAY_LEN	32

typedef struct{
	size_t mem_free;
	int next_id;				//id for next instance
	int pnum;				//number of processes
	size_t claimed[GMM_MU_ARRAY_LEN];	//memory usage: f(id) -> index
	size_t in_gpu[GMM_MU_ARRAY_LEN];	
	size_t in_use[GMM_MU_ARRAY_LEN];	//in_gpu_mem >= in_use_mem
	size_t swapped[GMM_MU_ARRAY_LEN];	//claimed == swapped + in_gpu_mem
	int wait[GMM_MU_ARRAY_LEN];		//waiting list: f(id) -> index: 0(false), id(true)
} *gmm_shared, gmm_shared_s;

typedef enum {
	in_gpu_mem = 0,
	swapped,
	in_use,
	unmalloced,
	outlaw
} objState;

typedef struct{
	void *key;	//generated randomly
	void *devPtr;	//pointer to object in device
	void *memPtr;	//pointer to object in main memory
	size_t size;	//size of the object in bytes
	objState state;
	int in_use_count; //TODO
} *gmm_obj, gmm_obj_s;

#define GMM_OBJ_SIZE sizeof(gmm_obj_s)
#define GMM_KEY_SIZE sizeof(void *)

typedef struct{
	pthread_mutex_t mutex;
	cfuhash_table_t *all;
	cfuhash_table_t *in_gpu_mem;
	cfuhash_table_t *in_use;
	cfuhash_table_t *swapped;
	cfuhash_table_t *unmalloced;
	int count;			//number of objects
} *gmm_local, gmm_local_s;

#define GMM_KERNEL_ARG_NUM_MAX	16
typedef struct{
	gmm_obj in_use[GMM_KERNEL_ARG_NUM_MAX];
	int in_use_num;	
} *gmm_args, gmm_args_s;

#endif
