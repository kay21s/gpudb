#ifndef _GMM_META_INTERFACE_H_
#define _GMM_META_INTERFACE_H_

#include <device_launch_parameters.h>
#include <cuda.h>

int gmm_init_attach(void);
int gmm_reclaim(void);
int gmm_attach(void);
int gmm_detach(void);
unsigned long int gmm_getFreeMem(void);
void gmm_setFreeMem(unsigned long int);
int gmm_getID(void);

void print_gmm_sdata(void);
CUresult gmm_cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, unsigned int, CUstream, void**, void **);

cudaError_t gmm_malloc(void **, size_t);
cudaError_t gmm_free(void *);
cudaError_t cudaSetupArgument(const void *, size_t, size_t);

#endif
