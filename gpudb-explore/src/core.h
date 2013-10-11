#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "list.h"
#include "spinlock.h"
#include "atomic.h"


// State of a device memory region
typedef enum region_state {
	STATE_ATTACHED = 0,		// object allocated with device memory
	STATE_DETACHED,			// object not attached with device memory
	STATE_FREEING,			// object being freed
	STATE_EVICTING,			// object being evicted
	STATE_EVICTED
} region_state_t;

// RW hint passed to a device memory region.
// TODO: RW hints can be very delicate. For example, if DB knows a kernel
// only modifies part of a region, it can pass GMM a RW hint with the range.
struct rwhint {
	int flags;
};

// Device memory block.
#define BLOCKSIZE		(4096 * 1024)
struct block {
	int dev_valid;			// if data copy on device is valid
	int swp_valid;			// if data copy in host swap buffer is valid
	struct spinlock lock;	// r/w lock
};

// Device memory region.
// A device memory region is a virtual memory area allocated by the user
// program through cudaMalloc. It is logically partitioned into an array of
// fixed-length device memory blocks. Due to lack of system support, all blocks
// must be attached/detached together. But the valid and dirty statuses of
// each block are maintained separately.
struct region {
	long size;				// size of the object in bytes
	void *dev_addr;			// device memory address
	void *swp_addr;			// host swap buffer address
	struct block *blocks;	// device memory blocks
	struct spinlock lock;	// the lock that protects memory object state
	region_state_t state;	// state of the object
	atomic_t pinned;		// atomic pin counter

	struct rwhint rwhint;	// rw hint

	struct list_head entry_alloced;		// linked to the list of allocated
	struct list_head entry_attached;	// linked to the list of attached
};

// Maximum number of kernel arguments that may be device memory pointers
#define NREFS		32

// A kernel argument that is a device memory pointer
struct dptr_arg {
	struct region *r;		// the region this argument points to
	unsigned long off;		// device pointer offset in the region
	int flags;
	void *dptr;				// the actual device memory address
	unsigned long argoff;	// this argument's offset in the argument stack
};

// Kernel callback structure
struct kcb {
	struct region *rgns[NREFS];	// Regions referenced by the kernel
	int nrgns;					// Number of regions referenced
};

// The local GMM context
struct gmm_context {
	struct spinlock lock;				// TODO: what's the use of this lock??
	latomic_t size_attached;			// Total size of attached mem regions
	struct list_head list_alloced;		// List of all allocated mem regions
	struct list_head list_attached;		// LRU list of attached mem regions
	struct spinlock lock_alloced;
	struct spinlock lock_attached;
	cudaStream_t stream_dma;			// The CUDA stream for DMA operations
	cudaStream_t stream_kernel;			// The CUDA stream for kernel launches
};

// A victim region for being evicted
struct victim {
	struct region *r;		// for a local victim
	int client;				// for a remote victim
	struct list_head entry;
};

#define MIN(x, y)	((x) < (y) ? (x) : (y))

#define NRBLOCKS(size)		(((size) + BLOCKSIZE - 1) / BLOCKSIZE)
#define BLOCKIDX(offset)	((unsigned long)(offset) / BLOCKSIZE)
#define BLOCKUP(offset)		((offset + BLOCKSIZE) / BLOCKSIZE * BLOCKSIZE)

// TODO: for region_pin, change client's size_detachable if pinned first;
// for region_unpin, change client's size_detachable if unpiined last.
#define region_pinned(r)	atomic_read(&(r)->pinned)


// Invalidate all blocks in a region.
static inline void region_inval(struct region *r, int swp)
{
	int i;

	if (swp) {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].swp_valid = 0;
	}
	else {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].dev_valid = 0;
	}
}

// Validate all blocks in a region.
static inline void region_valid(struct region *r, int swp)
{
	int i;

	if (swp) {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].swp_valid = 1;
	}
	else {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].dev_valid = 1;
	}
}

// Whether pointer p is included in pointer array a[0:n)
static inline int is_included(void **a, int n, void *p)
{
	int i;

	for (i = 0; i < n; i++)
		if (a[i] == p)
			return 1;

	return 0;
}


// Functions exposed by GMM core
int gmm_context_init();
void gmm_context_fini();

cudaError_t gmm_cudaMalloc(void **devPtr, size_t size);
cudaError_t gmm_cudaFree(void *devPtr);
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset);
cudaError_t gmm_cudaMemcpyHtoD(
		void *dst,
		const void *src,
		size_t count);
cudaError_t gmm_cudaMemcpyDtoH(
		void *dst,
		const void *src,
		size_t count);
cudaError_t gmm_cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream);
cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count);
cudaError_t gmm_cudaLaunch(const char* entry);

#endif
