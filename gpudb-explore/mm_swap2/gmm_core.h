#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <builtin_types.h>
#include <cuda.h>
#include "./gmm_type.h"

/*********************************** SHM ***************************************/
#define GMM_SHARED	9006
#define GMM_SHARED_SIZE	sizeof(gmm_shared_s)

#define GMM_SEM_NAME	"_gmm_semaphore4_"

/*********************************** MISC ***************************************/
#define GMM_DEBUG_MODE

#ifdef GMM_DEBUG_MODE
	#define GMM_DEBUG(call)	call
	#define GMM_DEBUG2(head, call)	do{fprintf(stderr, head); fprintf(stderr, "\t"); call;} while(0)
#else
	#define GMM_DEBUG(call)	
	#define GMM_DEBUG2(head, call)
#endif

#define GET_TIMEVAL(_t) (_t.tv_sec + _t.tv_usec / 1000000.0)

#define GMM_IGNORE_SIZE 5038400

/*********************************** CUDA ***************************************/
#define CUDA_CU_PATH    "/usr/local/cuda-5.0/lib64/libcuinj64.so"
#define CUDA_CURT_PATH  "/usr/local/cuda-5.0/lib64/libcudart.so"

#define CUDA_SAFE_CALL_NO_SYNC(call) do {       \
        cudaError_t err = call;                 \
        if( cudaSuccess != err) {               \
                fprintf(stderr, "[mm] Cuda error in file '%s' in line %i : %d.\n",      \
                        __FILE__, __LINE__, err );                                      \
                exit(EXIT_FAILURE);                                                     \
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
#define INTERCEPT_CU(func, var)                                    \
  do {                                                          \
    if(var) break;                                              \
    void *__handle = dlopen(CUDA_CU_PATH, RTLD_LOCAL | RTLD_LAZY);                                 \
    var = (typeof(var)) (uintptr_t) dlsym(__handle, func);      \
    TREAT_ERROR();                                              \
  } while(0)

#define INTERCEPT_CUDA(func, var)       \
do {                                    \
        if(var) break;                  \
        void *__handle = RTLD_NEXT;     \
        var = (typeof(var)) (uintptr_t) dlsym(__handle, func);  \
        TREAT_ERROR();                  \
} while(0)

#define INTERCEPT_CUDA2(func, var)	\
do {                                    \
        if(var) break;                  \
        void *__handle = dlopen(CUDA_CURT_PATH, RTLD_LOCAL | RTLD_LAZY);                                 \
        var = (typeof(var)) (uintptr_t) dlsym(__handle, func);  \
        TREAT_ERROR();                  \
} while(0)


#endif
