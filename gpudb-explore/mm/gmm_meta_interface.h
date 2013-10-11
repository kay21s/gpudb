#ifndef _GMM_META_INTERFACE_H_
#define _GMM_META_INTERFACE_H_

#include <device_launch_parameters.h>

int gmm_init_attach(unsigned long int);
int gmm_reclaim(void);
int gmm_attach(void);
int gmm_detach(void);
int gmm_getMC(void);
void gmm_setMC(unsigned long int);

cudaError_t gmm_malloc(void **, size_t);
cudaError_t gmm_free(void *);

#endif
