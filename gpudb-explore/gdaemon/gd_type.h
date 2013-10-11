#ifndef _GD_TYPE_H_
#define _GD_TYPE_H_

#include <device_launch_parameters.h>

#define CUDAMALLOC 1
#define CUDAFREE 2
#define CUDAMEMCPY 3
#define CUDACONFIGURE 4
#define CUDALAUNCH 5
#define CUDASETUP 6
#define CUDASYNC 7
#define CUDAMEMSET 8

typedef struct {
	int type;
	void *args;
} cudaCall, *cudaCall_sp;

typedef struct {
	void **devPtr;
	size_t size;
} argMalloc, *argMalloc_sp;

typedef struct {
	void *devPtr;
} argFree, *argFree_sp;

typedef struct {
	void *dst; 
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind;
} argMemcpy, *argMemcpy_sp;

typedef struct {
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
} argConfigureCall, *argConfigureCall_sp;

typedef struct {
	const char *entry;
} argLaunch, *argLaunch_sp;

typedef struct {
	const void *arg;
	size_t size;
	size_t offset;
} argSetup, *argSetup_sp;

typedef struct {
	void* devPtr;
	int value;
	size_t count;
} argMemset, *argMemset_sp;

#endif
