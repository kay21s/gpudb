#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>

#include "common.h"
#include "client.h"
#include "core.h"
#include "hint.h"
#include "replacement.h"
#include "msq.h"


// CUDA function handlers, defined in gmm_interfaces.c
extern cudaError_t (*nv_cudaMalloc)(void **, size_t);
extern cudaError_t (*nv_cudaFree)(void *);
//extern cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t,
//		enum cudaMemcpyKind);
extern cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *,
		size_t, enum cudaMemcpyKind, cudaStream_t stream);
extern cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *);
extern cudaError_t (*nv_cudaStreamDestroy)(cudaStream_t);
extern cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t);
extern cudaError_t (*nv_cudaMemGetInfo)(size_t*, size_t*);
extern cudaError_t (*nv_cudaSetupArgument) (const void *, size_t, size_t);
extern cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
extern cudaError_t (*nv_cudaMemset)(void * , int , size_t );
//extern cudaError_t (*nv_cudaMemsetAsync)(void * , int , size_t, cudaStream_t);
//extern cudaError_t (*nv_cudaDeviceSynchronize)(void);
extern cudaError_t (*nv_cudaLaunch)(const void *);
extern cudaError_t (*nv_cudaStreamAddCallback)(cudaStream_t,
		cudaStreamCallback_t, void*, unsigned int);

// TODO: declare all internal functions here, otherwise gcc
// might export those symbols.
static int gmm_free(struct region *m);
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count);
static int gmm_dtoh(
		struct region *r,
		void *dst,
		const void *src,
		size_t count);
static int gmm_load(struct region **rgns, int nrgns);
static int gmm_launch(const char *entry, struct region **rgns, int nrgns);
static struct region *region_lookup(struct gmm_context *ctx, const void *ptr);

// The GMM context for this process
struct gmm_context *pcontext = NULL;


static void list_alloced_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_add(&r->entry_alloced, &ctx->list_alloced);
	release(&ctx->lock_alloced);
}

static void list_alloced_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_del(&r->entry_alloced);
	release(&ctx->lock_alloced);
}

static void list_attached_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_add(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}

static void list_attached_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_del(&r->entry_attached);
	release(&ctx->lock_attached);
}

static void list_attached_mov(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_move(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}


static inline void region_pin(struct region *r)
{
	int pinned = atomic_inc(&(r)->pinned);
	if (pinned == 0)
		update_detachable(-r->size);
}

static inline void region_unpin(struct region *r)
{
	int pinned = atomic_dec(&(r)->pinned);
	if (pinned == 1)
		update_detachable(r->size);
}

// Initialize local GMM context.
int gmm_context_init()
{
	if (pcontext != NULL) {
		GMM_DPRINT("pcontext already exists!\n");
		return -1;
	}

	pcontext = (struct gmm_context *)malloc(sizeof(*pcontext));
	if (!pcontext) {
		GMM_DPRINT("failed to malloc for pcontext: %s\n", strerror(errno));
		return -1;
	}

	initlock(&pcontext->lock);		// ???
	latomic_set(&pcontext->size_attached, 0L);
	INIT_LIST_HEAD(&pcontext->list_alloced);
	INIT_LIST_HEAD(&pcontext->list_attached);
	initlock(&pcontext->lock_alloced);
	initlock(&pcontext->lock_attached);

	if (nv_cudaStreamCreate(&pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("failed to create DMA stream\n");
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	if (nv_cudaStreamCreate(&pcontext->stream_kernel) != cudaSuccess) {
		GMM_DPRINT("failed to create kernel stream\n");
		nv_cudaStreamDestroy(pcontext->stream_dma);
		free(pcontext);
		pcontext = NULL;
		return -1;
	}

	return 0;
}

void gmm_context_fini()
{
	struct list_head *p;
	struct region *r;

	// Free all dangling memory regions.
	while (!list_empty(&pcontext->list_alloced)) {
		p = pcontext->list_alloced.next;
		r = list_entry(p, struct region, entry_alloced);
		if (!gmm_free(r))
			list_move_tail(p, &pcontext->list_alloced);
	}

	nv_cudaStreamDestroy(pcontext->stream_dma);
	nv_cudaStreamDestroy(pcontext->stream_kernel);
	free(pcontext);
	pcontext = NULL;
}

// Allocate a new device memory object.
// We only allocate the host swap buffer space for now, and return
// the address of the host buffer to the user as the identifier of
// the object.
cudaError_t gmm_cudaMalloc(void **devPtr, size_t size)
{
	struct region *r;
	int nblocks;

	if (size > memsize_total()) {
		GMM_DPRINT("cudaMalloc size (%lu) too large (max %ld)", \
				size, memsize_total());
		return cudaErrorInvalidValue;
	}

	r = (struct region *)malloc(sizeof(*r));
	if (!r) {
		GMM_DPRINT("malloc for a new region: %s\n", strerror(errno));
		return cudaErrorMemoryAllocation;
	}
	memset(r, 0, sizeof(*r));

	r->swp_addr = malloc(size);
	if (!r->swp_addr) {
		GMM_DPRINT("malloc failed for host swap buffer: %s\n", strerror(errno));
		free(r);
		return cudaErrorMemoryAllocation;
	}

	nblocks = NRBLOCKS(size);
	r->blocks = (struct block *)malloc(sizeof(struct block) * nblocks);
	if (!r->blocks) {
		GMM_DPRINT("malloc failed for blocks array: %s\n", strerror(errno));
		free(r->swp_addr);
		free(r);
		return cudaErrorMemoryAllocation;
	}
	memset(r->blocks, 0, sizeof(struct block) * nblocks);

	// TODO: test how CUDA runtime align the size of device memory allocations
	r->size = (long)size;
	initlock(&r->lock);
	r->state = STATE_DETACHED;
	atomic_set(&r->pinned, 0);
	r->rwhint.flags = HINT_DEFAULT;

	list_alloced_add(pcontext, r);
	*devPtr = r->swp_addr;
	return cudaSuccess;
}

cudaError_t gmm_cudaFree(void *devPtr)
{
	struct region *r;

	if (!(r = region_lookup(pcontext, devPtr))) {
		GMM_DPRINT("could not find memory region containing %p\n", devPtr);
		// XXX: this is a workaround for regions allocated by CUDA runtime.
		// Return success even if devPtr is not found. FIXME!
		return cudaSuccess;
		//return cudaErrorInvalidDevicePointer;
	}

	if (gmm_free(r) < 0)
		return cudaErrorUnknown;
	else
		return cudaSuccess;
}

cudaError_t gmm_cudaMemcpyHtoD(
		void *dst,
		const void *src,
		size_t count)
{
	struct region *r;

	if (count <= 0)
		return cudaErrorInvalidValue;

	r = region_lookup(pcontext, dst);
	if (!r) {
		GMM_DPRINT("cannot find device memory region containing %p\n", dst);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING) {
		GMM_DPRINT("region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (dst + count > r->swp_addr + r->size) {
		GMM_DPRINT("htod device memory access out of boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_htod(r, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

cudaError_t gmm_cudaMemcpyDtoH(
		void *dst,
		const void *src,
		size_t count)
{
	struct region *r;

	if (count <= 0)
		return cudaErrorInvalidValue;

	r = region_lookup(pcontext, src);
	if (!r) {
		GMM_DPRINT("cannot find device memory region containing %p\n", src);
		return cudaErrorInvalidDevicePointer;
	}
	if (r->state == STATE_FREEING) {
		GMM_DPRINT("region already freed\n");
		return cudaErrorInvalidValue;
	}
	if (src + count > r->swp_addr + r->size) {
		GMM_DPRINT("dtoh device memory access out of boundary\n");
		return cudaErrorInvalidValue;
	}

	if (gmm_dtoh(r, dst, src, count) < 0)
		return cudaErrorUnknown;

	return cudaSuccess;
}

cudaError_t gmm_cudaMemGetInfo(size_t *free, size_t *total)
{
	*free = (size_t)memsize_free();
	*total = (size_t)memsize_total();
	return cudaSuccess;
}

// Which stream is the upcoming kernel to be issued to?
static cudaStream_t stream_issue = 0;

// TODO: Currently, %stream_issue is always set to pcontext->stream_kernel.
// This is not the best solution because it forbids kernels from being
// issued to different streams, which is required for, e.g., concurrent
// kernel executions.
// A better design is to prepare a kernel callback queue in pcontext->kcb
// for each possible stream ; kernel callbacks are registered in queues where
// they are issued to. This both maintains the correctness of kernel callbacks
// and retains the capability that kernels being issued to multiple streams.
cudaError_t gmm_cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	stream_issue = pcontext->stream_kernel;
	return nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream_issue);
}

// Reference hints passed for a kernel launch. Set by
// cudaReference in interfaces.c.
extern int refs[NREFS];
extern int rwflags[NREFS];
extern int nrefs;

// The device pointer arguments in the following kernel to be launched.
// TODO: should prepare the following structures for each stream.
static struct dptr_arg dargs[NREFS];
static int nargs = 0;
static int iarg = 0;

// CUDA pushes kernel arguments from left to right. For example, for a kernel
//				k(a, b, c)
// , a will be pushed on top of the stack, followed by b, and finally c.
// %offset gives the actual offset of an argument in the call stack,
// rather than which argument is being pushed.
// Let's assume cudaSetupArgument is invoked following the sequence of
// arguments from left to right.
cudaError_t gmm_cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{
	struct region *r;
	cudaError_t ret;
	int is_dptr = 0;
	int i = 0;

	// Test whether this argument is a device memory pointer.
	// If it is, record it and postpone its pushing until cudaLaunch.
	// Use reference hints if given. Otherwise, parse automatically
	// (but parsing errors are possible, e.g., when the user passes a
	// long argument that happen to lay within some region's host swap
	// buffer area).
	// XXX: we should assume all memory regions are to be referenced
	// if no reference hints are given.
	if (nrefs > 0) {
		for (i = 0; i < nrefs; i++) {
			if (refs[i] == iarg)
				break;
		}
		if (i < nrefs) {
			if (size != sizeof(void *))
				panic("cudaSetupArgument does not match cudaReference");
			r = region_lookup(pcontext, *(void **)arg);
			if (!r)
				// TODO: report error more gracefully
				panic("region_lookup in cudaSetupArgument");
			is_dptr = 1;
		}
	}
	else if (size == sizeof(void *)) {
		r = region_lookup(pcontext, *(void **)arg);
		if (r)
			is_dptr = 1;
	}

	if (is_dptr) {
		dargs[nargs].r = r;
		dargs[nargs].off = (unsigned long)(*(void **)arg - r->swp_addr);
		if (nrefs > 0)
			dargs[nargs].flags = rwflags[i];
		else
			dargs[nargs].flags = HINT_DEFAULT;
		dargs[nargs++].argoff = offset;
		ret = cudaSuccess;
	}
	else
		// This argument is not a device memory pointer.
		// XXX: Currently we ignore the case that nv_cudaSetupArgument
		// returns error and CUDA runtime might stop pushing arguments.
		ret = nv_cudaSetupArgument(arg, size, offset);

	iarg++;
	return ret;
}

// XXX: we should assume all memory regions are to be
// referenced if no reference hints are given.
// I.e., if (nargs == 0) do the following; else add all regions;
static long regions_referenced(struct region ***prgns, int *pnrgns)
{
	struct region **rgns;
	long total = 0;
	int nrgns = 0;
	int i;

	if (nrefs > NREFS)
		panic("nrefs");
	if (nargs < 0 || nargs > NREFS)
		panic("nargs");

	rgns = (struct region **)malloc(sizeof(*rgns) * nargs);
	if (!rgns) {
		GMM_DPRINT("malloc failed for region array: %s\n", strerror(errno));
		return -1;
	}

	for (i = 0; i < nargs; i++) {
		if (!is_included((void **)rgns, nrgns, (void*)(dargs[i].r))) {
			rgns[nrgns++] = dargs[i].r;
			dargs[i].r->rwhint.flags = dargs[i].flags;
			total += dargs[i].r->size;
		}
		else
			dargs[i].r->rwhint.flags |= dargs[i].flags;
	}

	*pnrgns = nrgns;
	if (nrgns > 0)
		*prgns = rgns;
	else {
		free(rgns);
		*prgns = NULL;
	}

	return total;
}

// Priority of the kernel launch (defined in interfaces.c).
// TODO: have to arrange something in global shared memory
// to expose kernel launch priority and scheduling info.
extern int prio_kernel;

// It is possible that multiple device pointers fall into
// the same region. So we first need to get the list of
// unique regions referenced by the kernel being launched.
// Then, load all referenced regions. At any time, only one
// context is allowed to load regions for kernel launch.
// This is to avoid deadlocks/livelocks caused by memory
// contentions from simultaneous kernel launches.
//
// TODO: Add kernel scheduling logic. The order of loadings
// plays an important role for fully overlapping kernel executions
// and DMAs. Ideally, kernels with nothing to load should be issued
// first. Then the loadings of other kernels can be overlapped with
// kernel executions.
// TODO: Maybe we should allow multiple loadings to happen
// simultaneously if we know the amount of free memory is enough.
cudaError_t gmm_cudaLaunch(const char *entry)
{
	cudaError_t ret = cudaSuccess;
	struct region **rgns = NULL;
	int nrgns = 0;
	long total = 0;
	int i, ldret;

	// NOTE: it is possible nrgns == 0 when regions_referenced
	// returns. Consider a kernel that only uses registers, for
	// example.
	total = regions_referenced(&rgns, &nrgns);
	if (total < 0 || total > memsize_total()) {
		GMM_DPRINT("kernel requires too much device memory space (%ld)\n", \
				total);
		ret = cudaErrorInvalidConfiguration;
		goto finish;
	}

reload:
	launch_wait();
	ldret = gmm_load(rgns, nrgns);
	launch_signal();
	if (ldret > 0) {	// load unsuccessful, retry later
		sched_yield();
		goto reload;
	}
	else if (ldret < 0) {	// fatal load error, quit launching
		GMM_DPRINT("load failed; quitting kernel launch\n");
		ret = cudaErrorUnknown;
		goto finish;
	}

	// Process RW hints.
	/// XXX: What if the launch below failed? Partial modification?
	for (i = 0; i < nrgns; i++) {
		if (rgns[i]->rwhint.flags & HINT_WRITE) {
			region_inval(rgns[i], 1);
			region_valid(rgns[i], 0);
		}
	}

	// Push all device pointer arguments.
	for (i = 0; i < nargs; i++) {
		dargs[i].dptr = dargs[i].r->dev_addr + dargs[i].off;
		nv_cudaSetupArgument(&dargs[i].dptr, sizeof(void *), dargs[i].argoff);
	}

	// Now we can launch the kernel.
	if (gmm_launch(entry, rgns, nrgns) < 0) {
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = cudaErrorUnknown;
	}

finish:
	if (rgns)
		free(rgns);
	nrefs = 0;
	nargs = 0;
	iarg = 0;
	return ret;
}

// TODO
cudaError_t gmm_cudaMemset(void * devPtr, int value, size_t count)
{
	return nv_cudaMemset(devPtr, value, count);
}

// The return value of this function tells whether the region has been
// immediately freed. 0 - not freed yet; 1 - freed.
static int gmm_free(struct region *r)
{
	// First, properly inspect/set region state.
re_acquire:
	acquire(&r->lock);
	switch (r->state) {
	case STATE_ATTACHED:
		if (!region_pinned(r))
			list_attached_del(pcontext, r);
		else {
			release(&r->lock);
			sched_yield();
			goto re_acquire;
		}
		break;
	case STATE_EVICTING:
		// Tell the evictor that this region is being freed
		r->state = STATE_FREEING;
		release(&r->lock);
		return 0;
	case STATE_FREEING:
		// The evictor has not seen this region being freed
		release(&r->lock);
		sched_yield();
		goto re_acquire;
	default: // STATE_DETACHED
		break;
	}
	release(&r->lock);

	// Now, this memory region can be freed.
	list_alloced_del(pcontext, r);
	if (r->blocks)
		free(r->blocks);
	if (r->swp_addr)
		free(r->swp_addr);
	if (r->dev_addr) {
		nv_cudaFree(r->dev_addr);
		latomic_sub(&pcontext->size_attached, r->size);
		update_attached(-r->size);
		update_detachable(-r->size);
	}
	free(r);

	return 1;
}

// TODO: provide two implementations - sync and async.
// For async, use streamcallback to unpin if necessary.
static int gmm_memcpy_dtoh(void *dst, const void *src, unsigned long size)
{
	if (nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
			pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("DtoH (%lu, %p => %p) failed\n", size, src, dst);
		return -1;
	}

	if (nv_cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// TODO: sync and async.
static int gmm_memcpy_htod(void *dst, const void *src, unsigned long size)
{
	if (nv_cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
			pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("DtoH (%lu, %p => %p) failed\n", size, src, dst);
		return -1;
	}

	if (nv_cudaStreamSynchronize(pcontext->stream_dma) != cudaSuccess) {
		GMM_DPRINT("Sync over DMA stream failed\n");
		return -1;
	}

	return 0;
}

// Sync the host and device copies of a data block.
// The direction of sync is determined by current valid flags. Data are synced
// from the valid copy to the invalid copy.
static void block_sync(struct region *r, int block)
{
	int dvalid = r->blocks[block].dev_valid;
	int svalid = r->blocks[block].swp_valid;
	unsigned long off, size;

	// Nothing to sync if both are valid or both are invalid
	if ((dvalid ^ svalid) == 0)
		return;
	if (!r->dev_addr || !r->swp_addr)
		panic("block_sync");

	off = block * BLOCKSIZE;
	size = MIN(off + BLOCKSIZE, r->size) - off;
	if (dvalid && !svalid) {
		// Sync from device to host swap buffer
		gmm_memcpy_dtoh(r->swp_addr + off, r->dev_addr + off, size);
		r->blocks[block].swp_valid = 1;
	}
	else {
		// Sync from host swap buffer to device
		gmm_memcpy_htod(r->dev_addr + off, r->swp_addr + off, size);
		r->blocks[block].dev_valid = 1;
	}
}

// Copy a piece of data from $src to (the host swap buffer of) a block in $r.
// $offset gives the offset of the destination address relative to $r->swp_addr.
// $size is the size of data to be copied.
// $block tells which block is being modified.
// $skip specifies whether to skip copying if the block is being locked.
// $skipped, if not null, returns whether skipped (1 - skipped; 0 - not skipped)
#ifdef GMM_CONFIG_HTOD_RADICAL
// This is an implementation matching the radical version of gmm_htod.
static void gmm_htod_block(
		struct region *r,
		unsigned long offset,
		const void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;

	// partial modification
	if ((offset % BLOCKSIZE) || (size < BLOCKSIZE && offset + size < r->size)) {
		if (b->swp_valid || !b->dev_valid) {
			// no locking needed
			memcpy(r->swp_addr + offset, src, size);
			if (!b->swp_valid)
				b->swp_valid = 1;
			if (b->dev_valid)
				b->dev_valid = 0;
		}
		else {
			// locking needed
			while (!try_acquire(&b->lock)) {
				if (skip) {
					if (skipped)
						*skipped = 1;
					return;
				}
			}
			if (b->swp_valid || !b->dev_valid) {
				release(&r->blocks[block].lock);
				memcpy(r->swp_addr + offset, src, size);
				if (!b->swp_valid)
					b->swp_valid = 1;
				if (b->dev_valid)
					b->dev_valid = 0;
			}
			else {
				// XXX: We don't need to pin the device memory because we are
				// holding the lock of a swp_valid=0,dev_valid=1 block, which
				// will prevent the evictor, if any, from freeing the device
				// memory under us.
				block_sync(r, block);
				release(&b->lock);
				memcpy(r->swp_addr + offset, src, size);
				b->dev_valid = 0;
			}
		}
	}
	// Full over-writing (its valid flags have been set in advance).
	else {
		while (!try_acquire(&b->lock)) {
			if (skip) {
				if (skipped)
					*skipped = 1;
				return;
			}
		}
		// Acquire the lock and release immediately, to avoid data races with
		// the evictor who happens to be writing the swp buffer.
		release(&b->lock);
		memcpy(r->swp_addr + offset, src, size);
	}

	if (skipped)
		*skipped = 0;
}
#else
// This is an implementation matching the conservative version of gmm_htod.
static void gmm_htod_block(
		struct region *r,
		unsigned long offset,
		const void *src,
		unsigned long size,
		int block,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + block;
	int partial = (offset % BLOCKSIZE) ||
			(size < BLOCKSIZE && offset + size < r->size);

	if (b->swp_valid || !b->dev_valid) {
		// no locking needed
		memcpy(r->swp_addr + offset, src, size);
		if (!b->swp_valid)
			b->swp_valid = 1;
		if (b->dev_valid)
			b->dev_valid = 0;
	}
	else {
		// locking needed
		while (!try_acquire(&b->lock)) {
			if (skip) {
				if (skipped)
					*skipped = 1;
				return;
			}
		}
		if (b->swp_valid || !b->dev_valid) {
			release(&r->blocks[block].lock);
			memcpy(r->swp_addr + offset, src, size);
			if (!b->swp_valid)
				b->swp_valid = 1;
			if (b->dev_valid)
				b->dev_valid = 0;
		}
		else {
			if (partial) {
				// XXX: We don't need to pin the device memory because we are
				// holding the lock of a swp_valid=0,dev_valid=1 block, which
				// will prevent the evictor, if any, from freeing the device
				// memory under us.
				block_sync(r, block);
				release(&b->lock);
			}
			else {
				b->swp_valid = 1;
				release(&b->lock);
			}
			memcpy(r->swp_addr + offset, src, size);
			b->dev_valid = 0;
		}
	}

	if (skipped)
		*skipped = 0;
}
#endif

// Handle a HtoD data transfer request.
// Note: the region may enter/leave STATE_EVICTING any time.
//
// Over-writing a whole block is different from modifying a block partially.
// The former can be handled by invalidating the dev copy of the block
// and setting the swp copy valid; the later requires a sync of the dev
// copy to the swp, if the dev has a newer copy, before the data can be
// written to the swp.
//
// Block-based memory management improves concurrency. Another important factor
// to consider is to reduce unnecessary swapping, if the region is being
// evicted during an HtoD action. Here we provide two implementations:
// one is radical, the other is conservative.
#if defined(GMM_CONFIG_HTOD_RADICAL)
// The radical version
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	int iblock, ifirst, ilast;
	unsigned long off, end;
	void *s = (void *)src;
	char *skipped;

	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// For each full-block over-writing, set dev_valid=0 and swp_valid=1.
	// Since we know the memory range being over-written, setting flags ahead
	// help prevent the evictor, if there is one, from wasting time evicting
	// those blocks. This is one unique advantage of us compared with CPU
	// memory management, where the OS usually does not have such interfaces
	// or knowledge.
	if (ifirst == ilast &&
		(count == BLOCKSIZE || (off == 0 && count == r->size))) {
		r->blocks[ifirst].dev_valid = 0;
		r->blocks[ifirst].swp_valid = 1;
	}
	else if (ifirst < ilast) {
		if (off % BLOCKSIZE == 0) {	// first block
			r->blocks[ifirst].dev_valid = 0;
			r->blocks[ifirst].swp_valid = 1;
		}
		if (end % BLOCKSIZE == 0 || end == r->size) {	// last block
			r->blocks[ilast].dev_valid = 0;
			r->blocks[ilast].swp_valid = 1;
		}
		for (iblock = ifirst + 1; iblock < ilast; iblock++) {	// the rest
			r->blocks[iblock].dev_valid = 0;
			r->blocks[iblock].swp_valid = 1;
		}
	}

	// Then, copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted). skipped[]
	// records whether a block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		gmm_htod_block(r, off, s, size, iblock, 1, skipped + (iblock - ifirst));
		s += size;
		off += size;
	}

	// Finally, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_htod_block(r, off, s, size, iblock, 0, NULL);
		s += size;
		off += size;
	}

	free(skipped);
	return 0;
}
#else
// The conservative version
static int gmm_htod(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	char *skipped;
	void *s = (void *)src;

	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted).
	// skipped[] records whether each block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		gmm_htod_block(r, off, s, size, iblock, 1, skipped + (iblock - ifirst));
		s += size;
		off += size;
	}

	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_htod_block(r, off, s, size, iblock, 0, NULL);
		s += size;
		off += size;
	}

	free(skipped);
	return 0;
}
#endif

static int gmm_dtoh_block(
		struct region *r,
		void *dst,
		unsigned long off,
		unsigned long size,
		int iblock,
		int skip,
		char *skipped)
{
	struct block *b = r->blocks + iblock;

	if (b->swp_valid) {
		memcpy(dst, r->swp_addr + off, size);
		return 0;
	}

	while (!try_acquire(&b->lock)) {
		if (skip) {
			if (skipped)
				*skipped = 1;
			return 0;
		}
	}

	if (b->swp_valid) {
		release(&b->lock);
		memcpy(dst, r->swp_addr + off, size);
	}
	else if (!b->dev_valid) {
		release(&b->lock);
	}
	else if (skip) {
		release(&b->lock);
		if (skipped)
			*skipped = 1;
		return 0;
	}
	else {
		// We don't need to pin the device memory because we are holding the
		// lock of a swp_valid=0,dev_valid=1 block, which will prevent the
		// evictor, if any, from freeing the device memory under us.
		block_sync(r, iblock);
		release(&b->lock);
		memcpy(dst, r->swp_addr + off, size);
	}

	if (skipped)
		*skipped = 0;

	return 0;
}

// TODO: It is possible to achieve pipelined copying, i.e., copy a block from
// its host swap buffer to user buffer while the next block is being fetched
// from device memory.
static int gmm_dtoh(
		struct region *r,
		void *dst,
		const void *src,
		size_t count)
{
	unsigned long off = (unsigned long)(src - r->swp_addr);
	unsigned long end = off + count, size;
	int ifirst = BLOCKIDX(off), iblock;
	char *skipped;

	skipped = (char *)malloc(BLOCKIDX(end - 1) - ifirst + 1);
	if (!skipped) {
		GMM_DPRINT("malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// First, copy blocks whose swp buffers contain immediate, valid data.
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		gmm_dtoh_block(r, dst, off, size, iblock, 1, skipped + iblock - ifirst);
		dst += size;
		off += size;
		iblock++;
	}

	// Then, copy the rest blocks.
	off = (unsigned long)(src - r->swp_addr);
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst])
			gmm_dtoh_block(r, dst, off, size, iblock, 0, NULL);
		dst += size;
		off += size;
		iblock++;
	}

	free(skipped);
	return 0;
}

// Look up a memory object by the ptr passed from user program.
// ptr should fall within the virtual memory area of the host swap buffer of
// the memory object, if it can be found.
static struct region *region_lookup(struct gmm_context *ctx, const void *ptr)
{
	struct region *r = NULL;
	struct list_head *pos;
	int found = 0;

	acquire(&ctx->lock_alloced);
	list_for_each(pos, &ctx->list_alloced) {
		r = list_entry(pos, struct region, entry_alloced);
		if ((unsigned long)ptr >= (unsigned long)r->swp_addr &&
			(unsigned long)ptr < ((unsigned long)r->swp_addr + r->size)) {
			found = 1;
			break;
		}
	}
	release(&ctx->lock_alloced);

	if (!found)
		r = NULL;

	return r;
}

// Select victims for %size_needed bytes of free device memory space.
// %excls[0:%nexcl) are local regions that should not be selected.
// Put selected victims in the list %victims.
int victim_select(
		long size_needed,
		struct region **excls,
		int nexcl,
		int local_only,
		struct list_head *victims)
{
	int ret = 0;

#if defined(GMM_REPLACEMENT_LRU)
	ret = victim_select_lru(size_needed, excls, nexcl, local_only, victims);
#elif defined(GMM_REPLACEMENT_LFU)
	ret = victim_select_lfu(size_needed, excls, nexcl, local_only, victims);
#else
	panic("replacement policy not specified");
	ret = -1;
#endif

	return ret;
}

// NOTE: When a local region is evicted, no other parties are
// supposed to be accessing the region at the same time.
// This is not true if multiple loadings happen simultaneously,
// but this region has been locked in region_load() anyway.
int region_evict(struct region *r)
{
	int nblocks = NRBLOCKS(r->size);
	char *skipped;
	int i;

	if (!r->dev_addr)
		panic("dev_addr is null");
	if (region_pinned(r))
		panic("evicting a pinned region");

	skipped = (char *)malloc(nblocks);
	if (!skipped) {
		GMM_DPRINT("malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// First round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto finish;
		if (try_acquire(&r->blocks[i].lock)) {
			if (!r->blocks[i].swp_valid)
				block_sync(r, i);
			release(&r->blocks[i].lock);
			skipped[i] = 0;
		}
		else
			skipped[i] = 1;
	}

	// Second round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto finish;
		if (skipped[i]) {
			acquire(&r->blocks[i].lock);
			if (!r->blocks[i].swp_valid)
				block_sync(r, i);
			release(&r->blocks[i].lock);
		}
	}

finish:
	list_attached_del(pcontext, r);
	nv_cudaFree(r->dev_addr);
	r->dev_addr = NULL;
	latomic_sub(&pcontext->size_attached, r->size);
	update_attached(-r->size);
	update_detachable(-r->size);
	region_inval(r, 0);
	acquire(&r->lock);
	if (r->state == STATE_FREEING) {
		if (r->swp_addr) {
			free(r->swp_addr);
			r->swp_addr = NULL;
		}
	}
	r->state = STATE_DETACHED;
	release(&r->lock);

	free(skipped);
	return 0;
}

// NOTE: Client %client should have been pinned when this function
// is called.
int remote_victim_evict(int client, long size_needed)
{
	int ret;
	ret = msq_send_req_evict(client, size_needed, 1);
	client_unpin(client);
	return ret;
}

// Similar to gmm_evict, but only select at most one victim from local
// region list, even if it is smaller than required, evict it, and return.
int local_victim_evict(long size_needed)
{
	struct list_head victims;
	struct victim *v;
	struct region *r;
	int ret;

	ret = victim_select(size_needed, NULL, 0, 1, &victims);
	if (ret != 0)
		return ret;

	if (list_empty(&victims))
		return 0;

	v = list_entry(victims.next, struct victim, entry);
	r = v->r;
	free(v);
	return region_evict(r);
}

// Evict the victim %victim.
// %victim may point to a local region or a remote client that
// may own some evictable region.
int victim_evict(struct victim *victim, long size_needed)
{
	if (victim->r)
		return region_evict(victim->r);
	else if (victim->client != -1)
		return remote_victim_evict(victim->client, size_needed);
	else {
		panic("victim is neither local nor remote");
		return -1;
	}
}

// Evict some device memory so that the size of free space can
// satisfy %size_needed. Regions in %excls[0:%nexcl) should not
// be selected for eviction.
static int gmm_evict(long size_needed, struct region **excls, int nexcl)
{
	struct list_head victims, *e;
	struct victim *v;
	int ret = 0;

	INIT_LIST_HEAD(&victims);

	do {
		ret = victim_select(size_needed, excls, nexcl, 0, &victims);
		if (ret != 0)
			return ret;

		for (e = victims.next; e != (&victims); ) {
			v = list_entry(e, struct victim, entry);
			if (memsize_free() < size_needed) {
				if ((ret = victim_evict(v, size_needed)) != 0)
					goto fail_evict;
			}
			else if (v->r) {
				acquire(&v->r->lock);
				if (v->r->state != STATE_FREEING)
					v->r->state = STATE_ATTACHED;
				release(&v->r->lock);
			}
			else if (v->client != -1)
				client_unpin(v->client);
			list_del(e);
			e = e->next;
			free(v);
		}
	} while (memsize_free() < size_needed);

	return 0;

fail_evict:
	for (e = victims.next; e != (&victims); ) {
		v = list_entry(e, struct victim, entry);
		if (v->r) {
			acquire(&v->r->lock);
			if (v->r->state != STATE_FREEING)
				v->r->state = STATE_ATTACHED;
			release(&v->r->lock);
		}
		else if (v->client != -1)
			client_unpin(v->client);
		list_del(e);
		e = e->next;
		free(v);
	}

	return ret;
}

// Allocate device memory to a region (i.e., attach).
static int region_attach(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	int ret;

	if (r->state != STATE_DETACHED) {
		GMM_DPRINT("nothing to attach\n");
		return -1;
	}

	// Attach if current free memory space is larger than region size.
	if (r->size <= memsize_free() &&
		nv_cudaMalloc(&r->dev_addr, r->size) == cudaSuccess)
		goto attach_success;

	// Evict some device memory.
	ret = gmm_evict(r->size, excls, nexcl);
	if (ret < 0 || (ret > 0 && memsize_free() < r->size))
		return ret;

	// Try to attach again.
	if (nv_cudaMalloc(&r->dev_addr, r->size) != cudaSuccess) {
		r->dev_addr = NULL;
		return 1;
	}

attach_success:
	latomic_add(&pcontext->size_attached, r->size);
	update_attached(r->size);
	update_detachable(r->size);
	if (pin)
		region_pin(r);
	// Reassure that the dev copies of all blocks are set to invalid.
	region_inval(r, 0);
	r->state = STATE_ATTACHED;
	list_attached_add(pcontext, r);

	return 0;
}

// Load a region to device memory.
// excls[0:nexcl) are regions that should not be evicted when
// evictions need to happen during the loading.
static int region_load(
		struct region *r,
		int pin,
		struct region **excls,
		int nexcl)
{
	int i, ret;

	if (r->state == STATE_EVICTING || r->state == STATE_FREEING) {
		GMM_DPRINT("should not see a evicting/freeing region during loading\n");
		return -1;
	}

	// Attach if the region is still detached.
	if (r->state == STATE_DETACHED) {
		if ((ret = region_attach(r, 1, excls, nexcl)) != 0)
			return ret;
	}
	else {
		if (pin)
			region_pin(r);
		// Update the region's position in the LRU list.
		list_attached_mov(pcontext, r);
	}

	// Fetch data to device memory if necessary.
	if (r->rwhint.flags & HINT_READ) {
		for (i = 0; i < NRBLOCKS(r->size); i++) {
			acquire(&r->blocks[i].lock);	// Though this is useless
			if (!r->blocks[i].dev_valid)
				block_sync(r, i);
			release(&r->blocks[i].lock);
		}
	}

	return 0;
}

// Load all %n regions specified by %rgns to device.
// Every successfully loaded region is pinned to device.
// If all regions cannot be loaded successfully, successfully
// loaded regions will be unpinned so that they can be
// replaced by other kernel launches.
// Return value: 0 - success; < 0 - fatal failure; > 0 - retry later.
static int gmm_load(struct region **rgns, int n)
{
	char *pinned;
	int i, ret;

	if (n == 0)
		return 0;
	if (n < 0 || (n > 0 && !rgns))
		return -1;

	pinned = (char *)malloc(n);
	if (!pinned) {
		GMM_DPRINT("malloc failed for pinned array: %s\n", strerror(errno));
		return -1;
	}
	memset(pinned, 0, n);

	for (i = 0; i < n; i++) {
		if (rgns[i]->state == STATE_FREEING) {
			GMM_DPRINT("warning: not loading freed region\n");
			continue;
		}
		// NOTE: In current design, this locking is redundant
		acquire(&rgns[i]->lock);
		ret = region_load(rgns[i], 1, rgns, n);
		release(&rgns[i]->lock);
		if (ret != 0)
			goto fail;
		pinned[i] = 1;
	}

	free(pinned);
	return 0;

fail:
	for (i = 0; i < n; i++)
		if (pinned[i])
			region_unpin(rgns[i]);
	free(pinned);
	return ret;
}

// The callback function invoked by CUDA after each kernel finishes
// execution. Have to keep it as short as possible because it blocks
// the following commands in the stream.
void CUDART_CB gmm_kernel_callback(
		cudaStream_t stream,
		cudaError_t status,
		void *data)
{
	struct kcb *pcb = (struct kcb *)data;
	int i;
	for (i = 0; i < pcb->nrgns; i++)
		region_unpin(pcb->rgns[i]);
	free(pcb);
}

// Here we utilize CUDA 5+'s stream callback feature to capture kernel
// finish event and unpin related regions accordingly.
static int gmm_launch(const char *entry, struct region **rgns, int nrgns)
{
	struct kcb *pcb;

	if (nrgns > NREFS) {
		GMM_DPRINT("too many regions\n");
		return -1;
	}

	pcb = (struct kcb *)malloc(sizeof(*pcb));
	if (!pcb) {
		GMM_DPRINT("malloc failed for kcb: %s\n", strerror(errno));
		return -1;
	}
	if (nrgns > 0)
		memcpy(&pcb->rgns, rgns, sizeof(void *) * nrgns);
	pcb->nrgns = nrgns;

	if (nv_cudaLaunch(entry) != cudaSuccess) {
		free(pcb);
		return -1;
	}
	nv_cudaStreamAddCallback(stream_issue, gmm_kernel_callback, (void *)pcb, 0);

	return 0;
}
