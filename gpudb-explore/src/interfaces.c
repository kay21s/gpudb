// The CUDA runtime interfaces that intercept/enhance the
// default CUDA runtime with DB resource management
// functionalities. All functions/symbols exported by the
// library reside in this file.

#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "core.h"
#include "client.h"
#include "protocol.h"
#include "interfaces.h"
#include "hint.h"


// Default CUDA runtime function handlers.
cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
cudaError_t (*nv_cudaFree)(void *) = NULL;
cudaError_t (*nv_cudaMemcpy)(void *, const void *,
		size_t, enum cudaMemcpyKind) = NULL;
cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *,
		size_t, enum cudaMemcpyKind, cudaStream_t stream) = NULL;
cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *) = NULL;
cudaError_t (*nv_cudaStreamDestroy)(cudaStream_t) = NULL;
cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t) = NULL;
cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*) = NULL;
cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t) = NULL;
cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
cudaError_t (*nv_cudaMemset)(void * , int , size_t ) = NULL;
//cudaError_t (*nv_cudaMemsetAsync)(void * , int , size_t, cudaStream_t) = NULL;
//cudaError_t (*nv_cudaDeviceSynchronize)(void) = NULL;
cudaError_t (*nv_cudaLaunch)(const void *) = NULL;
cudaError_t (*nv_cudaStreamAddCallback)(cudaStream_t,
		cudaStreamCallback_t, void*, unsigned int) = NULL;

static int initialized = 0;


// The library constructor.
// The order of initialization matters. First, link to the default
// CUDA interface implementations, because CUDA interfaces have
// been intercepted by our library and we should be able to redirect
// CUDA calls to their default implementations if GMM environment
// fails to initialize successfully. Then, initialize GMM local
// context. Finally, after the local environment has been initialized
// successfully, we connect our local context to the global GMM arena
// and let the party begin.
__attribute__((constructor))
void gmm_init(void)
{
	INTERCEPT_CUDA2("cudaMalloc", nv_cudaMalloc);
	INTERCEPT_CUDA2("cudaFree", nv_cudaFree);
	INTERCEPT_CUDA2("cudaMemcpy", nv_cudaMemcpy);
	INTERCEPT_CUDA2("cudaMemcpyAsync", nv_cudaMemcpyAsync);
	INTERCEPT_CUDA2("cudaStreamCreate", nv_cudaStreamCreate);
	INTERCEPT_CUDA2("cudaStreamDestroy", nv_cudaStreamDestroy);
	INTERCEPT_CUDA2("cudaStreamSynchronize", nv_cudaStreamSynchronize);
	INTERCEPT_CUDA2("cudaMemGetInfo", nv_cudaMemGetInfo);
	INTERCEPT_CUDA2("cudaSetupArgument", nv_cudaSetupArgument);
	INTERCEPT_CUDA2("cudaConfigureCall", nv_cudaConfigureCall);
	INTERCEPT_CUDA2("cudaMemset", nv_cudaMemset);
	//INTERCEPT_CUDA2("cudaMemsetAsync", nv_cudaMemsetAsync);
	//INTERCEPT_CUDA2("cudaDeviceSynchronize", nv_cudaDeviceSynchronize);
	INTERCEPT_CUDA2("cudaLaunch", nv_cudaLaunch);
	INTERCEPT_CUDA2("cudaStreamAddCallback", nv_cudaStreamAddCallback);

	if (gmm_context_init() == -1) {
		GMM_DPRINT("failed to initialize GMM local context\n");
		return;
	}

	if (client_attach() == -1) {
		GMM_DPRINT("failed to attach to the GMM global arena\n");
		gmm_context_fini();
		return;
	}

	// Before marking GMM context initialized, invoke an NV function
	// to initialize CUDA runtime and let whatever memory regions
	// implicitly required by CUDA runtime be allocated now. Those
	// regions should be always attached and not managed by GMM runtime.
	do {
		size_t dummy;
		nv_cudaMemGetInfo(&dummy, &dummy);
	} while (0);

	initialized = 1;
	GMM_DPRINT("gmm initialized\n");
}

// The library destructor.
__attribute__((destructor))
void gmm_fini(void)
{
	if (initialized) {
		client_detach();
		gmm_context_fini();
		initialized = 0;
		GMM_DPRINT("gmm finished\n");
	}
}

GMM_EXPORT
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaMalloc(devPtr, size);
	else {
		// TODO: we need to remember those device memory allocated
		// before GMM was initialized, so that later when they are
		// used in cudaMemcpy or other functions we can treat them
		// specially.
		GMM_DPRINT("warning: cudaMalloc called outside of GMM\n");
		ret = nv_cudaMalloc(devPtr, size);
	}

	return ret;
}

// GMM-specific: allowing passing read/write hints.
//GMM_EXPORT
//cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags)
//{
//	if (initialized)
//		return gmm_cudaMalloc(devPtr, size, flags);
//	else
//		return cudaErrorInitializationError;
//}

GMM_EXPORT
cudaError_t cudaFree(void *devPtr)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaFree(devPtr);
	else {
		GMM_DPRINT("warning: cudaFree called outside of GMM\n");
		ret = nv_cudaFree(devPtr);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaMemcpy(
		void *dst,
		const void *src,
		size_t count,
		enum cudaMemcpyKind kind)
{
	cudaError_t ret;

	if (initialized) {
		if (kind == cudaMemcpyHostToDevice)
			ret = gmm_cudaMemcpyHtoD(dst, src, count);
		else
			ret = gmm_cudaMemcpyDtoH(dst, src, count);
	}
	else {
		GMM_DPRINT("warning: cudaMemcpy called outside of GMM\n");
		ret = nv_cudaMemcpy(dst, src, count, kind);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaMemGetInfo(free, total);
	else
		ret = nv_cudaMemGetInfo(free, total);

	return ret;
}

GMM_EXPORT
cudaError_t cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	else {
		GMM_DPRINT("warning: cudaConfigureCall called outside of GMM\n");
		ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaSetupArgument(arg, size, offset);
	else {
		GMM_DPRINT("warning: cudaSetupArgument called outside of GMM\n");
		ret = nv_cudaSetupArgument(arg, size, offset);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaMemset(void * devPtr, int value, size_t count)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaMemset(devPtr, value, count);
	else {
		GMM_DPRINT("warning: cudaMemset called outside of GMM\n");
		ret = nv_cudaMemset(devPtr, value, count);
	}

	return ret;
}

//GMM_EXPORT
//cudaError_t cudaDeviceSynchronize()
//{
//	return nv_cudaDeviceSynchronize();
//}

GMM_EXPORT
cudaError_t cudaLaunch(const void *entry)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaLaunch(entry);
	else {
		GMM_DPRINT("warning: cudaLaunch called outside of GMM\n");
		ret = nv_cudaLaunch(entry);
	}

	return ret;
}

// The priority of the next kernel launch.
// Value ranges from 0 (highest) to PRIO_MAX (lowest).
int prio_kernel = PRIO_DEFAULT;

// GMM-specific: specify kernel launch priority.
GMM_EXPORT
cudaError_t cudaSetKernelPrio(int prio)
{
	if (!initialized)
		return cudaErrorInitializationError;
	if (prio < 0 || prio > PRIO_LOWEST)
		return cudaErrorInvalidValue;

	prio_kernel = prio;
	return cudaSuccess;
}

// For passing reference hints before each kernel launch.
// TODO: should prepare the following structures for each stream.
int refs[NREFS];
int rwflags[NREFS];
int nrefs = 0;

// GMM-specific: pass reference hints.
// %which_arg tells which argument (starting with 0) in the following
// cudaSetupArgument calls is a device memory pointer. %flags is the
// read-write flag.
// The GMM runtime should expect to see call sequence similar to below:
// cudaReference, ..., cudaReference, cudaConfigureCall,
// cudaSetupArgument, ..., cudaSetupArgument, cudaLaunch
//
GMM_EXPORT
cudaError_t cudaReference(int which_arg, int flags)
{
	int i;

	if (!initialized)
		return cudaErrorInitializationError;

	if (which_arg < NREFS) {
		for (i = 0; i < nrefs; i++) {
			if (refs[i] == which_arg)
				break;
		}
		if (i == nrefs) {
			refs[nrefs] = which_arg;
			rwflags[nrefs++] = flags;
		}
		else
			rwflags[i] |= flags;
	}
	else {
		GMM_DPRINT("bad cudaReference argument %d (max %d)\n", \
				which_arg, NREFS-1);
		return cudaErrorInvalidValue;
	}

	return cudaSuccess;
}

