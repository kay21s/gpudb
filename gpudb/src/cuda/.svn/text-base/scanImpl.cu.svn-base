#ifndef SCAN_IMPL_CU
#define SCAN_IMPL_CU

#include "scan.cu"
#include "../include/common.h"

static void scanImpl(int *d_input, int rLen, int *d_output, struct statistic * pp)
{
	preallocBlockSums(rLen);
	prescanArray(d_output, d_input, rLen, pp);
	deallocBlockSums();
}


#endif

