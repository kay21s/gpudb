#ifndef _GMM_CORE_INTERFACE_H_
#define _GMM_CORE_INTERFACE_H_

#include <device_launch_parameters.h>
#include <cuda.h>

int gmm_init_attach(void);
int gmm_reclaim(void);
int gmm_attach(void);
int gmm_detach(void);
size_t gmm_getFreeMem(void);
void gmm_setFreeMem(size_t);
int gmm_getID(void);

void gmm_print_sdata(void);

#endif
