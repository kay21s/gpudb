#ifndef _GMM_TYPE_FUNC_H_
#define _GMM_TYPE_FUNC_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "./gmm_type.h"

#define GMM_DDEBUG_MODE

#ifdef GMM_DDEBUG_MODE
        #define GMM_DDEBUG(call) call
#else
        #define GMM_DDEBUG(call)
#endif

/********************************** shared data **************************************/
// by default, shmget initialize all values to 0
inline int init_gmm_shared(gmm_shared s, size_t free){ 
	(s)->mem_free = free;
	(s)->next_id = 1;
	(s)->pnum = 0;

	return 0;
}

inline int new_gmm_id(gmm_shared s) {
	return s->next_id++;
}

inline int clean_gmm_shared(gmm_shared s, int id) {
	size_t claimed = s->claimed[id % GMM_MU_ARRAY_LEN];
	//assume swapped == 0
	s->mem_free += claimed;
	s->claimed[id % GMM_MU_ARRAY_LEN] = 0;
	s->in_gpu[id % GMM_MU_ARRAY_LEN] = 0;
	s->in_use[id % GMM_MU_ARRAY_LEN] = 0;
	s->swapped[id % GMM_MU_ARRAY_LEN] = 0;
	s->wait[id % GMM_MU_ARRAY_LEN] = 0;	//false(0)

	return 0;
}

inline size_t get_gmm_shared_free(gmm_shared s) {
	return s->mem_free;
}

inline int set_gmm_shared_free(gmm_shared s, size_t free){ 
	s->mem_free = free;

	return 0;
}

#define S_GET_MEM_FREE(_ptr_)	(_ptr_->mem_free)
#define S_GET_MEM_CLAIMED(_ptr_, _id)	((_ptr_->claimed[_id % GMM_MU_ARRAY_LEN]))
#define S_GET_MEM_IN_GPU(_ptr_, _id)	((_ptr_->in_gpu[_id % GMM_MU_ARRAY_LEN]))
#define S_GET_MEM_IN_USE(_ptr_, _id)	((_ptr_->in_use[_id % GMM_MU_ARRAY_LEN]))
#define S_GET_MEM_SWAPPED(_ptr_, _id)	((_ptr_->swapped[_id % GMM_MU_ARRAY_LEN]))
#define S_IS_WAIT(_ptr_, _id_)   ((_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN] != 0))
#define S_SET_WAIT(_ptr_, _id_)   ((_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN] = _id_))
#define S_RESET_WAIT(_ptr_, _id_)   ((_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN] = 0))

#define S_SET_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _size_;} while(0)
#define S_INC_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _ptr_->mem_free + _size_;} while(0)
#define S_DEC_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _ptr_->mem_free - _size_;} while(0)
#define S_INC_MEM_IN_GPU(_ptr_, _id, _size_)	do {_ptr_->in_gpu[_id % GMM_MU_ARRAY_LEN] = _ptr_->in_gpu[_id % GMM_MU_ARRAY_LEN] + _size_;} while(0)
#define S_DEC_MEM_IN_GPU(_ptr_, _id, _size_)	do {_ptr_->in_gpu[_id % GMM_MU_ARRAY_LEN] = _ptr_->in_gpu[_id % GMM_MU_ARRAY_LEN] - _size_;} while(0)
#define S_INC_MEM_SWAPPED(_ptr_, _id, _size_)	do {_ptr_->swapped[_id % GMM_MU_ARRAY_LEN] = _ptr_->swapped[_id % GMM_MU_ARRAY_LEN] + _size_;} while(0)
#define S_DEC_MEM_SWAPPED(_ptr_, _id, _size_)	do {_ptr_->swapped[_id % GMM_MU_ARRAY_LEN] = _ptr_->swapped[_id % GMM_MU_ARRAY_LEN] - _size_;} while(0)
#define S_INC_MEM_IN_USE(_ptr_, _id, _size_)	do {_ptr_->in_use[_id % GMM_MU_ARRAY_LEN] = _ptr_->in_use[_id % GMM_MU_ARRAY_LEN] + _size_;} while(0)
#define S_DEC_MEM_IN_USE(_ptr_, _id, _size_)	do {_ptr_->in_use[_id % GMM_MU_ARRAY_LEN] = _ptr_->in_use[_id % GMM_MU_ARRAY_LEN] - _size_;} while(0)
#define S_INC_MEM_CLAIMED(_ptr_, _id, _size_)	do {_ptr_->claimed[_id % GMM_MU_ARRAY_LEN] = _ptr_->claimed[_id % GMM_MU_ARRAY_LEN] + _size_;} while(0)
#define S_DEC_MEM_CLAIMED(_ptr_, _id, _size_)	do {_ptr_->claimed[_id % GMM_MU_ARRAY_LEN] = _ptr_->claimed[_id % GMM_MU_ARRAY_LEN] - _size_;} while(0)

#define S_NOT_SHARED(_ptr_)	((_ptr_->pnum == 1))
#define S_INC_PNUM(_ptr_)	do {_ptr_->pnum++;} while(0)
#define S_DEC_PNUM(_ptr_)	do {_ptr_->pnum--;} while(0)

inline int update_gmm_shared_remove_obj(gmm_shared s, int id, size_t size, objState state) {
	if (state == in_gpu_mem) {
		S_DEC_MEM_IN_GPU(s, id, size);
		S_INC_MEM_FREE(s, size);
	} else if (state == swapped) {
		S_DEC_MEM_SWAPPED(s, id, size);
	} else if (state == in_use) {
		S_DEC_MEM_IN_USE(s, id, size);
		S_INC_MEM_FREE(s, size);
	} else if (state == outlaw) {
		S_INC_MEM_FREE(s, size);
	}

	S_DEC_MEM_CLAIMED(s, id, size);

	return 0;
}

inline int update_gmm_shared_mv_obj(gmm_shared s, int id, size_t size, objState from, objState to) {
	if (from == in_gpu_mem) {
		S_DEC_MEM_IN_GPU(s, id, size);
		//S_INC_MEM_FREE(s, size);
	} else if (from == swapped) {
		S_DEC_MEM_SWAPPED(s, id, size);
	} else if (from == in_use) {
		S_DEC_MEM_IN_USE(s, id, size);
		//S_INC_MEM_FREE(s, size);
	}

	if (to == in_gpu_mem) {
		S_INC_MEM_IN_GPU(s, id, size);
		//S_DEC_MEM_FREE(s, size);
	} else if (to == swapped) {
		S_INC_MEM_SWAPPED(s, id, size);
	} else if (to == in_use) {
		S_INC_MEM_IN_USE(s, id, size);
		//S_DEC_MEM_FREE(s, size);
	}

	return 0;
}

#define S_MV_OBJ_GPU_MAIN(_s, _id, _size)	update_gmm_shared_mv_obj(_s, _id, _size, in_gpu_mem, swapped)
#define S_MV_OBJ_MAIN_GPU(_s, _id, _size)	update_gmm_shared_mv_obj(_s, _id, _size, swapped, in_gpu_mem)
#define S_MV_OBJ_GPU_USE(_s, _id, _size)	update_gmm_shared_mv_obj(_s, _id, _size, in_gpu_mem, in_use)
#define S_MV_OBJ_USE_GPU(_s, _id, _size)	update_gmm_shared_mv_obj(_s, _id, _size, in_use, in_gpu_mem)

inline void print_gmm_shared_info(gmm_shared s, int i) {
	fprintf(stderr, "[MEM]\tindex: %d\tclaimed: %lu\tin_gpu: %lu\tin_use: %lu\tswapped: %lu\twait: %d\n",
		i, S_GET_MEM_CLAIMED(s, i), S_GET_MEM_IN_GPU(s, i), S_GET_MEM_IN_USE(s, i), S_GET_MEM_SWAPPED(s, i), S_IS_WAIT(s, i));
	return;	
}

inline void print_gmm_shared(gmm_shared s) {
        int i = 0;
	size_t sum = 0;
        for (i = 0; i < GMM_MU_ARRAY_LEN; i++) {
                if (S_GET_MEM_CLAIMED(s, i) > 0 || S_IS_WAIT(s, i)) {
			print_gmm_shared_info(s, i);
			sum = sum + S_GET_MEM_CLAIMED(s, i) - S_GET_MEM_SWAPPED(s, i);
		}
        }
        fprintf(stderr, "[GMM_SUMMARY]: free: %lu\tclaimed: %lu\tall: %lu\n", get_gmm_shared_free(s), sum, get_gmm_shared_free(s)+sum);
}

inline int get_min_wait(gmm_shared g, int id) {
	int min = id;
	int i;
	for(i = 0; i < GMM_MU_ARRAY_LEN; i++)		
		if(g->wait[i]<min && g->wait[i]>0)	
			min = g->wait[i];			
	return min;
}

inline int get_max_wait(gmm_shared g) {
	int max = 0;
	int i;
	for(i = 0; i<GMM_MU_ARRAY_LEN; i++)		
		if(g->wait[i] > max && g->in_gpu[i] > 0)
			max = g->wait[i];	
	return max;
}


/********************************** local data **************************************/
#define GMM_LOCAL_HASH_SIZE 16384
#define PUT_HASH(_ht, _key, _obj, _objr)			\
	cfuhash_put_data(_ht, ((void*) &(_key)), GMM_KEY_SIZE,	\
		((void *)_obj), GMM_OBJ_SIZE, ((void**) &(_objr)));
#define DEL_HASH(_ht, _key)	cfuhash_delete_data(_ht, ((void *) &(_key)), GMM_KEY_SIZE);	

inline long lrand() {
    if (sizeof(int) < sizeof(long))
        return (((long)rand()) << (sizeof(int) * 8)) | rand();

    return rand();
}

#define WAIT_LOCAL_SEM(_l)	do {GMM_DDEBUG(fprintf(stderr, ">>>Claim LocalSem\n")); pthread_mutex_lock(&(_l->mutex) );} while(0)
#define POST_LOCAL_SEM(_l)	do {GMM_DDEBUG(fprintf(stderr, ">>>Release LocalSem\n")); pthread_mutex_unlock(&(_l->mutex) );} while(0)

/***************************************/

inline int init_gmm_local(gmm_local *lptr) {
	*lptr = (gmm_local)malloc(sizeof(gmm_local_s));
	gmm_local l = *lptr;

	if (pthread_mutex_init(&(l->mutex), NULL) != 0) {
		fprintf(stderr, "mutex init failed\n");
                exit(-1);
	}

	l->all = cfuhash_new_with_initial_size(GMM_LOCAL_HASH_SIZE);	
	l->in_gpu_mem = cfuhash_new_with_initial_size(GMM_LOCAL_HASH_SIZE);	
	l->in_use = cfuhash_new_with_initial_size(GMM_LOCAL_HASH_SIZE);	
	l->swapped = cfuhash_new_with_initial_size(GMM_LOCAL_HASH_SIZE);	
	l->unmalloced = cfuhash_new_with_initial_size(GMM_LOCAL_HASH_SIZE);	
	
	l->count = 0;
	srand(127);

	GMM_DDEBUG(fprintf(stderr, "[init]\tall_ptr: %p\tin_gpu_ptr: %p\n", l->all, l->in_gpu_mem) );
	//GMM_DDEBUG(cfuhash_pretty_print(l->in_gpu_mem, stderr));

	return 0;
}

inline int destroy_gmm_local(gmm_local l) {
	if (l != NULL) {
		pthread_mutex_destroy(&(l->mutex));
		free(l);
	}
	return 0;
}

inline void *create_add_obj(gmm_local l, void* devPtr, size_t size, objState state) {
	gmm_obj obj = (gmm_obj) malloc(GMM_OBJ_SIZE);
	gmm_obj objr = NULL; 
	if (state == outlaw)
		obj->key = devPtr;
	else
		obj->key = (void*) lrand();
	obj->devPtr = devPtr;
	obj->memPtr = NULL;
	obj->size = size;
	obj->state = state;
	int ret = 0;
	
	GMM_DDEBUG(fprintf(stderr, "[new obj]\tdevPtr: %p\tsize: %lu\tstate: %d\tkey: %p\tkeySize: %lu\tobjSize: %lu\taddr: %p\n", 
			devPtr, size, state, obj->key, GMM_KEY_SIZE, GMM_OBJ_SIZE, l->in_gpu_mem) );
	WAIT_LOCAL_SEM(l);
	if (state == in_gpu_mem) {
		ret = PUT_HASH(l->in_gpu_mem, obj->key, obj, objr);
	} else if (state == unmalloced) {
		ret = PUT_HASH(l->unmalloced, obj->key, obj, objr);
	}
	ret = PUT_HASH(l->all, obj->key, obj, objr);
	l->count ++;
	POST_LOCAL_SEM(l);
	
	return obj->key;
}

#define NEW_IN_GPU_OBJ(_l, _devPtr, _size)	\
	create_add_obj(_l, _devPtr, _size, in_gpu_mem)

#define NEW_UNMALLOC_OBJ(_l, _devPtr, _size)	\
	create_add_obj(_l, _devPtr, _size, unmalloced)

#define NEW_OUTLAW_OBJ(_l, _devPtr, _size)	\
	create_add_obj(_l, _devPtr, _size, outlaw)

//inline gmm_obj get_obj(gmm_local l, void *key) {
//	gmm_obj obj = NULL;
//	size_t obj_size = GMM_OBJ_SIZE;
//	int ret = cfuhash_get_data(l->all, (void*) &key, GMM_KEY_SIZE,
//		(void **) &obj, &obj_size);	
//
//	return obj;
//}

inline gmm_obj get_obj(gmm_local l, void *vkey) {
	void* key = NULL;
        size_t key_size = GMM_KEY_SIZE;
        gmm_obj obj = NULL;
        size_t obj_size = GMM_OBJ_SIZE;

	WAIT_LOCAL_SEM(l);
	int ret = cfuhash_get_data(l->all, (void*) &vkey, GMM_KEY_SIZE,
		(void **) &obj, &obj_size);
	if (obj != NULL && obj->state != outlaw) {
		POST_LOCAL_SEM(l);
		return obj;
	}	

        gmm_obj ret_obj = NULL;
	int hasNext = cfuhash_each_data(l->all, &key, &key_size, (void**)(&obj), &obj_size);
	while (hasNext) {
		GMM_DDEBUG(fprintf(stderr, "[get_obj_maybe]\tvkey: %p\tkey: %p\tsize: %lu\n", vkey, key, obj->size) );
		if ( obj->state != outlaw && ((size_t)vkey - (size_t)obj->key) < obj->size ) {
			GMM_DDEBUG(fprintf(stderr, "[get_obj_success]\tvkey: %p\tkey: %p\tsize: %lu\n", vkey, key, obj->size) );
			ret_obj = obj;
			break;
		}
		hasNext = cfuhash_next_data(l->all, &key, &key_size, (void**)(&obj), &obj_size);
	}
	POST_LOCAL_SEM(l);

	return ret_obj;
}

inline size_t get_obj_size(gmm_local l, void *key) {
	gmm_obj obj = get_obj(l, key);

	return obj->size;
}

inline objState get_obj_state(gmm_local l, void *key) {
	gmm_obj obj = get_obj(l, key);

	return obj->state;
}

inline void* get_obj_devPtr(gmm_local l, void *key) {
	gmm_obj obj = get_obj(l, key);

	return obj->devPtr + ((size_t)key - (size_t)obj->key);
}


inline int obj_exists(gmm_local l, void *key) {
	if (key == NULL)
		return 0;

	gmm_obj obj = get_obj(l, key);
	return (obj !=NULL);
}

inline gmm_obj get_one_obj(gmm_local l, objState state) {
	void* key = NULL;
	size_t key_size = GMM_KEY_SIZE;
	gmm_obj obj = NULL;
	size_t obj_size = GMM_OBJ_SIZE;
	
	WAIT_LOCAL_SEM(l);
	if (state == in_gpu_mem)
		cfuhash_each_data(l->in_gpu_mem, &key, &key_size, (void **)(&obj), &obj_size);
	else if (state == swapped)
		cfuhash_each_data(l->swapped, &key, &key_size, (void**)(&obj), &obj_size);
	POST_LOCAL_SEM(l);

	return obj;
}

inline int is_obj_in_x(gmm_local l, void *key, objState x) {
	gmm_obj obj = get_obj(l, key);
	if (obj->state == x)
		return 1;
	else
		return 0;
}

#define IS_OBJ_IN_GPU(_l, _k)	is_obj_in_x(_l, _k, in_gpu_mem)
#define IS_OBJ_IN_USE(_l, _k)	is_obj_in_x(_l, _k, in_use)

#define GET_ONE_IN_GPU_OBJ(_l)	get_one_obj(_l, in_gpu_mem)
#define GET_ONE_SWAPPED_OBJ(_l)	get_one_obj(_l, swapped)

inline int mv_obj_o(gmm_local l, gmm_obj obj, objState from, objState to) {
	gmm_obj objr;
	if (obj == NULL) {
		fprintf(stderr, "[error] mv_obj:Failed\n");
		return 1;
	}
	
	WAIT_LOCAL_SEM(l);
	if (from == in_gpu_mem) {
		DEL_HASH(l->in_gpu_mem, obj->key);
	} else if (from == swapped) {
		DEL_HASH(l->swapped, obj->key);
	} else if (from == in_use) {
		DEL_HASH(l->in_use, obj->key);
	}

	if (to == in_gpu_mem) {
		PUT_HASH(l->in_gpu_mem, obj->key, obj, objr);
	} else if (to == swapped) {
		PUT_HASH(l->swapped, obj->key, obj, objr);
	} else if (to == in_use) {
		PUT_HASH(l->in_use, obj->key, obj, objr);
	}
	obj->state = to;
	POST_LOCAL_SEM(l);

	return 0;
}

inline int mv_obj(gmm_local l, void* key, objState from, objState to) {
	return mv_obj_o(l, get_obj(l,key), from, to);
}

#define MV_OBJ_GPU_MAIN(_l, _key)	mv_obj(_l, _key, in_gpu_mem, swapped)
#define MV_OBJ_MAIN_GPU(_l, _key)	mv_obj(_l, _key, swapped, in_gpu_mem)
#define MV_OBJ_GPU_USE(_l, _key)	mv_obj(_l, _key, in_gpu_mem, in_use)
#define MV_OBJ_USE_GPU(_l, _key)	mv_obj(_l, _key, in_use, in_gpu_mem)

inline int delete_obj(gmm_local l, void *key) {
	WAIT_LOCAL_SEM(l);
	DEL_HASH(l->all, key);
	DEL_HASH(l->in_gpu_mem, key);
	DEL_HASH(l->in_use, key);
	DEL_HASH(l->swapped, key);
	(l->count)--;
	POST_LOCAL_SEM(l);
	
	return 0;
}

#define HAS_SWAPPED_OBJ(_l)	((cfuhash_num_entries(_l) > 0))

inline void print_gmm_obj(gmm_obj obj) {
        fprintf(stderr, "[OBJ]\tkey: %p\tdevPtr: %p\tmemPtr: %p\tsize: %lu\tstate: %d\n",
                obj->key, obj->devPtr, obj->memPtr, obj->size, obj->state);
        return;
}

inline void print_gmm_obj2(gmm_local l, void *key) {
	gmm_obj obj;
	size_t obj_size;
	if (!obj_exists(l, key)) {
        	fprintf(stderr, "[OBJ]\tkey %p does not exist\n", key);
		return;
	}
	int ret = cfuhash_get_data(l->all, (void*) &key, GMM_KEY_SIZE,
		(void **) &obj, &obj_size);	
        fprintf(stderr, "[OBJ]\tkey: %p\tdevPtr: %p\tmemPtr: %p\tsize: %lu\tstate: %d\n",
                obj->key, obj->devPtr, obj->memPtr, obj->size, obj->state);
        return;
}

inline int cfu_print_gmm_obj(void *key, size_t key_size, void *data, size_t data_size, void *arg) {
	print_gmm_obj((gmm_obj) data);
	return 0;
}

inline void print_gmm_local(gmm_local l) {
	WAIT_LOCAL_SEM(l);
	fprintf(stderr, "[gmm_local START all]\tcount: %d\n", l->count);
	cfuhash_foreach(l->all, cfu_print_gmm_obj, NULL);
	fprintf(stderr, "[gmm_local START in_gpu_mem]\n");
	cfuhash_foreach(l->in_gpu_mem, cfu_print_gmm_obj, NULL);
	fprintf(stderr, "[gmm_local START in_use]\n");
	cfuhash_foreach(l->in_use, cfu_print_gmm_obj, NULL);
	fprintf(stderr, "[gmm_local START swapped]\n");
	cfuhash_foreach(l->swapped, cfu_print_gmm_obj, NULL);
	fprintf(stderr, "[gmm_local START unmalloced]\n");
	cfuhash_foreach(l->unmalloced, cfu_print_gmm_obj, NULL);
	fprintf(stderr, "[gmm_local END]\n");
	POST_LOCAL_SEM(l);
        return;
}

inline void print_ptrs(void** base, size_t size) {
	int i = 0;
	fprintf(stderr, "[ptrs]\t base: %p\tsize: %lu\t", base, size);
	for(i = 0; i < size/8; i++) {
		fprintf(stderr, "[%d] %p\t", i, base[i]);
	}
	fprintf(stderr, "\n");
}

inline void print_dim3(dim3 vec) {
	fprintf(stderr, "[dim3] x: %u\ty: %u\tz: %u\n", vec.x, vec.y, vec.z);
}


/****************** Asynchronous Function Call --- Arguments ********************/
#define NEW_GMM_ARGS()			((gmm_args)calloc(sizeof(gmm_args_s), 1))
#define RESET_GMM_ARGS(_ptr)		(memset((void *)_ptr, 0, sizeof(gmm_args_s))	)
#define CPY_GMM_ARGS(_dst, _src)	do {memcpy((void*)_dst, (const void*)_src, sizeof(gmm_args_s));} while(0)
#define FREE_GMM_ARGS(_ptr)		(free(_ptr) )
#define GMM_ARGS_ADD_OBJ(_ptr, _obj)	do {_ptr->in_use[_ptr->in_use_num] = _obj; _ptr->in_use_num ++;} while(0)

inline void print_args(gmm_args p) {
	int i = 0;
	fprintf(stderr, "[args]\tnum: %lu\t", p->in_use_num);
	for(i=0; i<p->in_use_num; i++) {
		gmm_obj obj = p->in_use[i];
		fprintf(stderr, "[%d] %p<%lu>\t", i, obj->key, obj->size);
	}
	fprintf(stderr, "\n");
}

inline size_t update_local_objs_by_args(gmm_local l, gmm_args args) {
	size_t ingpu = 0;
	int i;
	void *key;

	for(i = 0; i < args->in_use_num; i++) {
		gmm_obj obj = args->in_use[i];
		// size of objs no longer in use
		ingpu += obj->size;
		MV_OBJ_USE_GPU(l, obj->key);
	}
	
	return ingpu;
}

#endif
