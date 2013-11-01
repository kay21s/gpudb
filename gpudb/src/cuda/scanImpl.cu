#ifndef SCAN_IMPL_CU
#define SCAN_IMPL_CU

#include "scan.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

static void scanImpl(int *d_input, int rLen, int *d_output, struct statistic * pp)
{
    int len = 2;
    if(rLen < len){
            int *input, *output;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&input,len*sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&output, len*sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemset(input, 0, len*sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(input, d_input, rLen*sizeof(int), cudaMemcpyDeviceToDevice));
            preallocBlockSums(len);
            prescanArray(output, input, len, pp);
            deallocBlockSums();
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_output,output,rLen*sizeof(int),cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(input));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(output));
            return;
    }else{
            preallocBlockSums(rLen);
            prescanArray(d_output, d_input, rLen, pp);
            deallocBlockSums();
    }
//	preallocBlockSums(rLen);
//	prescanArray(d_output, d_input, rLen, pp);
//	deallocBlockSums();
}


#endif

