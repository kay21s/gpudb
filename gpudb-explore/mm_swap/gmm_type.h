#ifndef _GMM_TYPE_H_
#define _GMM_TYPE_H_


// by default, shmget initialize all values to 0
#define GMM_MU_ARRAY_LEN	32
#define GMM_IGNORE_SIZE	16384
#define GMM_KERNEL_PARAM_LEN_MAX 16

typedef struct{
	unsigned long int mem_free;
	int next_id;
	unsigned long int mu[GMM_MU_ARRAY_LEN];	//memory usage: f(id) -> index
	int wait[GMM_MU_ARRAY_LEN];		//waiting list: f(id) -> index: 0(false), id(true)
} *gmm_shared, gmm_shared_s;

#define INIT_GMM_SHARED(_ptr_, _mem_free_) do {_ptr_->mem_free = _mem_free_; _ptr_->next_id = 1;} while(0)	
#define NEW_GMM_ID(_ptr_)	(_ptr_->next_id++)

#define GET_MU(_ptr_, _id_)	(_ptr_->mu[_id_ % GMM_MU_ARRAY_LEN])
#define INC_MU(_ptr_, _id_, _size_)	do{_ptr_->mu[_id_ % GMM_MU_ARRAY_LEN] += _size_;}while(0)
#define DEC_MU(_ptr_, _id_, _size_)	do{_ptr_->mu[_id_ % GMM_MU_ARRAY_LEN] -= _size_;}while(0)

#define GET_WAIT(_ptr_, _id_)	(_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN])
#define SET_WAIT(_ptr_, _id_)	do{_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN]=_id_;}while(0)
#define RESET_WAIT(_ptr_, _id_)	do{_ptr_->wait[_id_ % GMM_MU_ARRAY_LEN]=0;}while(0)
#define GET_MIN_WAIT(_ptr, _min, _id, _i_)	\
	do { 					\
		for(_i_=0,_min=_id;_i_<GMM_MU_ARRAY_LEN; _i_++)		\
			if(_ptr->wait[_i_]<_min && _ptr->wait[_i_]>0)	\
				_min = _ptr->wait[_i_];			\
	} while(0)

#define GET_MAX_WAIT(_ptr, _max, _i_)		\
	do { 					\
		for(_i_=0,_max=0; _i_<GMM_MU_ARRAY_LEN; _i_++)		\
			if(_ptr->wait[_i_] > _max && _ptr->mu[_i_] > 0)	\
				_max = _ptr->wait[_i_];			\
	} while(0)

#define GET_MEM_FREE(_ptr_)	(_ptr_->mem_free)
#define SET_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _size_;} while(0)
#define DEC_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _ptr_->mem_free - _size_;} while(0)
#define INC_MEM_FREE(_ptr_, _size_)	do {_ptr_->mem_free = _ptr_->mem_free + _size_;} while(0)


#define LOCAL_HASH_SIZE 262147
typedef struct{
	void **ptr;	//first arguement of cudaMalloc
	void *devPtr;	//pointer to object in device
	void *memPtr;	//pointer to object in main memory
	size_t size;	//size of the object in bytes
} *gmm_obj, gmm_obj_s;

typedef struct{
	gmm_obj_s objs[LOCAL_HASH_SIZE];
	int next[LOCAL_HASH_SIZE];	//index of next non-empty object, tail->next == -1
	int prev[LOCAL_HASH_SIZE];	
	int head;			//index of the "first" non-empty object
	int count;			//number of objects
	int swapped;			//number of objects swapped
} *gmm_local, gmm_local_s;

#define MALLOC_LOCAL(_l)	do {_l = (gmm_local)malloc(sizeof(gmm_local_s)); _l->head = -1;} while(0)

#define ADD_OBJ(_l, _f, _ptrPtr, _devPtr, _size)	\
	do { 						\
		(_l->objs[_f(_devPtr)]).ptr = (void**) _ptrPtr;		\
		(_l->objs[_f(_devPtr)]).devPtr = (void*) _devPtr;	\
		(_l->objs[_f(_devPtr)]).memPtr = NULL;	\
		(_l->objs[_f(_devPtr)]).size = _size;	\
		if(_l->head != -1) _l->prev[_l->head] = _f(_devPtr);	\
		_l->next[_f(_devPtr)] = _l->head;	\
		_l->prev[_f(_devPtr)] = -1;		\
		_l->head = _f(_devPtr);			\
		_l->count += 1;				\
	} while(0)

#define ADD_OBJ2(_l, _f, _ptrPtr, _devPtr, _memPtr, _size)		\
	do { 								\
		(_l->objs[_f(_devPtr)]).ptr = (void**) _ptrPtr;		\
		(_l->objs[_f(_devPtr)]).devPtr = (void*) _devPtr;	\
		(_l->objs[_f(_devPtr)]).memPtr = (void*)_memPtr;	\
		(_l->objs[_f(_devPtr)]).size = _size;			\
		if(_l->head != -1) _l->prev[_l->head] = _f(_devPtr);	\
		_l->next[_f(_devPtr)] = _l->head;	\
		_l->prev[_f(_devPtr)] = -1;		\
		_l->head = _f(_devPtr);			\
		_l->count += 1;				\
	} while(0)

#define RESET_OBJ(_l, _f, _ptrPtr, _devPtr, _memPtr, _size)		\
	do { 								\
		(_l->objs[_f(_devPtr)]).ptr = (void**) _ptrPtr;		\
		(_l->objs[_f(_devPtr)]).devPtr = (void*) _devPtr;	\
		(_l->objs[_f(_devPtr)]).memPtr = (void*)_memPtr;	\
		(_l->objs[_f(_devPtr)]).size = _size;			\
	} while(0)

#define GET_OBJ_SIZE(_l, _f, _devPtr)	((_l->objs[_f(_devPtr)]).size)
#define GET_OBJ(_l, _f, _devPtr)	(&(_l->objs[_f(_devPtr)]))
#define OBJ_EXISTS(_l, _f, _devPtr)	(((_l->objs[_f(_devPtr)]).ptr != NULL))

#define DEL_OBJ(_l, _f, _devPtr)	\
	do {				\
					\
		if ((_l->objs[_f(_devPtr)]).ptr != NULL) {		\
			if(_l->next[_f(_devPtr)] != -1){				\
				_l->prev[_l->next[_f(_devPtr)]] = _l->prev[_f(_devPtr)];\
			}		\
			if(_l->prev[_f(_devPtr)] != -1){				\
				_l->next[_l->prev[_f(_devPtr)]] = _l->next[_f(_devPtr)];\
			}else{						\
				_l->head = _l->next[_f(_devPtr)];	\
			}						\
			(_l->objs[_f(_devPtr)]).ptr = NULL;		\
			_l->count -= 1;					\
		}			\
	} while(0)

#define GET_SWAPPED(_l)	((_l->swapped))

#define CUDA_SAFE_CALL_NO_SYNC(call) do {	\
	cudaError_t err = call;			\
	if( cudaSuccess != err) {		\
		fprintf(stderr, "[mm] Cuda error in file '%s' in line %i : %d.\n",	\
			__FILE__, __LINE__, err );					\
		exit(EXIT_FAILURE);							\
	}} while(0)

/* check whether dlsym returned successfully */
#define  TREAT_ERROR()                          \
  do {                                          \
    char * __error;                             \
    if ((__error = dlerror()) != NULL)  {       \
      fputs(__error, stderr);                   \
      abort();                                  \
    }                                           \
  }while(0)

/* interception function func and store its previous value into var */
#define CUDA_CU_PATH	"/usr/local/cuda-5.0/lib64/libcuinj64.so"
#define CUDA_CURT_PATH	"/usr/local/cuda-5.0/lib64/libcudart.so"

#define INTERCEPT_CU(func, var)                                    \
  do {                                                          \
    if(var) break;                                              \
    void *__handle = dlopen(CUDA_CU_PATH, RTLD_LOCAL | RTLD_LAZY);                                 \
    var = (typeof(var)) (uintptr_t) dlsym(__handle, func);      \
    TREAT_ERROR();                                              \
  } while(0)

#define INTERCEPT_CUDA(func, var)	\
do {					\
	if(var) break;			\
	void *__handle = RTLD_NEXT;	\
	var = (typeof(var)) (uintptr_t) dlsym(__handle, func);	\
	TREAT_ERROR();			\
} while(0)

#define INTERCEPT_CUDA2(func, var)       \
do {                                    \
        if(var) break;                  \
    	void *__handle = dlopen(CUDA_CURT_PATH, RTLD_LOCAL | RTLD_LAZY);                                 \
        var = (typeof(var)) (uintptr_t) dlsym(__handle, func);  \
        TREAT_ERROR();                  \
} while(0)

#endif
