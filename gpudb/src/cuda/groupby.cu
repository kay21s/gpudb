/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <time.h>
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#include "../include/cudaHash.h"
#include "scanImpl.cu"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

/*
 * Transform integer to string using one single gpu thread.
 */

__device__ static char * gpuItoa(int value, char* result, int base){

        if (base < 2 || base > 36) {
                *result = '\0';
                return result;
        }

        char* ptr = result, *ptr1 = result, tmp_char;
        int tmp_value;

        do {
                tmp_value = value;
                value /= base;
                *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
        } while ( value );

        if (tmp_value < 0)
                *ptr++ = '-';

        *ptr-- = '\0';

        while(ptr1 < ptr) {
                tmp_char = *ptr;
                *ptr--= *ptr1;
                *ptr1++ = tmp_char;
        }
        return result;

}

/*
 * string copy using one gpu thread.
 */

__device__ static char * gpuStrcpy(char * dst, const char * src){

    char * orig = dst;
    while(*src)
            *dst++ = *src++;
    *dst = '\0';

    return orig;
}

__device__ static char* gpuStrncat(char *dest, const char *src, size_t n)
{
    int dest_len = 0;
    int i;

    char * tmp = dest;
    while(*tmp != '\0'){
        tmp++;
        dest_len ++;
    }

    for (i = 0 ; i < n && src[i] != '\0' ; i++)
        dest[dest_len + i] = src[i];
    dest[dest_len + i] = '\0';
    return dest;
}

__device__ static char * gpuStrcat(char * dest, const char * src){
    char *tmp =dest;
    int dest_len = 0;
    int i;

    while (*tmp!= '\0'){
        tmp++ ;
        dest_len ++;
    }

    for(i=0; src[i] !='\0'; i++){
        dest[dest_len + i] = src[i];
    }

    dest[dest_len + i] = '\0';

    return dest;
}

/*
 * Combine the group by columns to build the group by keys. 
 */

__global__ static void build_groupby_key(char ** content, int gbColNum, int * gbIndex, int * gbType, int * gbSize, long tupleNum, int * key, int *num){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(long i = offset; i< tupleNum; i+= stride){
        char buf[128] = {0};
        for (int j=0; j< gbColNum; j++){
            char tbuf[32]={0};
            int index = gbIndex[j];

            if (index == -1){
                gpuItoa(1,tbuf,10);
                gpuStrncat(buf,tbuf,1);

            }else if (gbType[j] == STRING){
                gpuStrncat(buf, content[index] + i*gbSize[j], gbSize[j]);

            }else if (gbType[j] == INT){
                int key = ((int *)(content[index]))[i];
                gpuItoa(key,tbuf,10);
                gpuStrcat(buf,tbuf);
            }
        }
        int hkey = StringHash(buf) % HSIZE;
        key[i]= hkey;
        num[hkey] = 1;
    }
}

/*
 * This is for testing only. 
 */

__global__ static void build_groupby_key_soa(char ** content, int gbColNum, int * gbIndex, int * gbType, int * gbSize, long tupleNum, int * key, int *num){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(long i = offset; i< tupleNum; i+= stride){
        char buf[128] = {0};
        for (int j=0; j< gbColNum; j++){
            char tbuf[32]={0};
            int index = gbIndex[j];

            if (index == -1){
                gpuItoa(1,tbuf,10);
                gpuStrncat(buf,tbuf,1);

            }else if (gbType[j] == STRING){
                for(int k=0;k<gbSize[j];k++){
                    long pos = k*tupleNum + i;
                    buf[k] = content[index][pos];
                }
                gpuStrncat(buf,tbuf,gbSize[j]);

            }else if (gbType[j] == INT){
                int key = ((int *)(content[index]))[i];
                gpuItoa(key,tbuf,10);
                gpuStrcat(buf,tbuf);
            }
        }
        int hkey = StringHash(buf) % HSIZE;
        key[i]= hkey;
        num[hkey] = 1;
    }
}


/*
 * Count the number of groups 
 */

__global__ void count_group_num(int *num, int tupleNum, int *totalCount){
        int stride = blockDim.x * gridDim.x;
        int offset = blockIdx.x * blockDim.x + threadIdx.x;
        int localCount = 0;

        for(int i=offset; i<tupleNum; i+= stride){
                if(num[i] == 1){
                        localCount ++;
                }
        }

        atomicAdd(totalCount,localCount);
}

/*
 * Calculate the groupBy expression.
 */

__device__ static float calMathExp(char **content, struct mathExp * exp, int pos, int op){
    float res ;

    if(op == NOOP){
        if (exp[0].opType == CONS)
            res = exp[0].opValue;
        else{
            int index = exp[0].opValue;
            res = ((int *)(content[index]))[pos];
        }
    
    }else if(op == PLUS ){
        res = calMathExp(content, &exp[0],pos, NOOP) + calMathExp(content, &exp[1], pos, NOOP);

    }else if (op == MINUS){
        res = calMathExp(content, &exp[0],pos, NOOP) - calMathExp(content, &exp[1], pos, NOOP);

    }else if (op == MULTIPLY){
        res = calMathExp(content, &exp[0],pos, NOOP) * calMathExp(content, &exp[1], pos, NOOP);

    }else if (op == DIVIDE){
        res = calMathExp(content, &exp[0],pos, NOOP) / calMathExp(content, &exp[1], pos, NOOP);
    }

    return res;
}

/*
 * group by constant. Currently only support SUM function.
 */

__global__ void agg_cal_cons(char ** content, int colNum, int * funcArray, int *op, struct mathExp* exp, int * mathOffset, int * gbType, int * gbSize, long tupleNum, int * key, int *psum,  char ** result){

    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[32];
    for(int i=0;i<32;i++)
        buf[i] = 0;

    for(int i=index;i<tupleNum;i+=stride){
        for(int j=0;j<colNum;j++){
            int func = funcArray[j];
            int offset = mathOffset[j];
            if (func == SUM){
                float tmpRes = calMathExp(content,&exp[offset] , i, op[j]);
                buf[j] += tmpRes;
            }
        }
    }

    for(int i=0;i<colNum;i++)
        atomicAdd(&((float *)result[i])[0], buf[i]);
}

/*
 * gropu by
 */

__global__ void agg_cal(char ** content, int colNum, int * funcArray, int * op, struct mathExp* exp, int * mathOffset, int * gbType, int * gbSize, long tupleNum, int * key, int *psum,  char ** result){

        int stride = blockDim.x * gridDim.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        for(int i=index;i<tupleNum;i+=stride){

            int hKey = key[i];
            int offset = psum[hKey];

            for(int j=0;j<colNum;j++){
                int func = funcArray[j];
                int mo = mathOffset[j];
                if(func ==NOOP){
                    int type = exp[mo].opType;

                    if(type == CONS){
                        int value = exp[mo].opValue;
                        ((int *)result[j])[offset] = value;
                    }else{
                        int index = exp[mo].opValue;
                        int attrSize = gbSize[j];
                        if(attrSize == sizeof(int))
                            ((int *)result[j])[offset] = ((int*)content[index])[i];
                        else
                            memcpy(result[j] + offset*attrSize, content[index] + i * attrSize, attrSize);
                    }

                }else if (func == SUM){
                    float tmpRes = calMathExp(content, &exp[mo], i, op[j]);
                    atomicAdd(& ((float *)result[j])[offset], tmpRes);
                }
            }
        }
}


/* 
 * groupBy: group by the data and calculate. 
 * 
 * Prerequisite:
 *  input data are not compressed
 *
 * Input:
 *  gb: the groupby node which contains the input data and groupby information
 *  pp: records the statistics such as kernel execution time 
 *
 * Return:
 *  a new table node
 */

struct tableNode * groupBy(struct groupByNode * gb, struct statistic * pp){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);
    int *gpuGbIndex = NULL, gpuTupleNum, gpuGbColNum;
    int *gpuGbType = NULL, *gpuGbSize = NULL;

    int *gpuGbKey = NULL;
    char ** gpuContent = NULL, **column = NULL;

    /*
     * @gbCount: the number of groups
     * gbConstant: whether group by constant
     */

    int gbCount;
    int gbConstant = 0;

    struct tableNode *res = (struct tableNode *) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->tupleSize = gb->tupleSize;
    res->totalAttr = gb->outputAttrNum;
    res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrType);
    res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrSize);
    res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrTotalSize);
    res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataPos);
    res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataFormat);
    res->content = (char **) malloc(sizeof(char **) * res->totalAttr);
    CHECK_POINTER(res->content);

    for(int i=0;i<res->totalAttr;i++){
        res->attrType[i] = gb->attrType[i];
        res->attrSize[i] = gb->attrSize[i];
        res->dataFormat[i] = UNCOMPRESSED;
    }
    
    gpuTupleNum = gb->table->tupleNum;
    gpuGbColNum = gb->groupByColNum;

    if(gpuGbColNum == 1 && gb->groupByIndex[0] == -1){
        gbConstant = 1;
    }


    dim3 grid(1024);
    dim3 block(128);
    int blockNum = gb->table->tupleNum / block.x + 1;
    if(blockNum < 1024)
        grid = blockNum;

    int *gpu_hashNum = NULL, *gpu_psum = NULL, *gpuGbCount = NULL;

#ifdef HAS_GMM
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocEx((void **)&gpuContent, gb->table->totalAttr * sizeof(char *), HINT_PTARRAY));
#else
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuContent, gb->table->totalAttr * sizeof(char *)));
#endif
    column = (char **) malloc(sizeof(char *) * gb->table->totalAttr);
    CHECK_POINTER(column);

    for(int i=0;i<gb->table->totalAttr;i++){
        int attrSize = gb->table->attrSize[i];
        if(gb->table->dataPos[i]==MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)& column[i], attrSize * gb->table->tupleNum));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], gb->table->content[i], attrSize *gb->table->tupleNum, cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
        }else{
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &gb->table->content[i], sizeof(char *), cudaMemcpyHostToDevice));
        }
    }

    if(gbConstant != 1){

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType,gb->groupByType, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbSize,gb->groupBySize, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));


        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbKey, gb->table->tupleNum * sizeof(int)));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbIndex, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbIndex, gb->groupByIndex,sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*HSIZE));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*HSIZE));

        do{
        	GMM_CALL(cudaReference(0, HINT_READ|HINT_PTARRAY|HINT_PTAREAD));
        	GMM_CALL(cudaReference(3, HINT_READ));
        	GMM_CALL(cudaReference(2, HINT_READ));
        	GMM_CALL(cudaReference(4, HINT_READ));
        	GMM_CALL(cudaReference(7, HINT_WRITE));
        	GMM_CALL(cudaReference(6, HINT_WRITE));
	        build_groupby_key<<<grid,block>>>(gpuContent,gpuGbColNum, gpuGbIndex, gpuGbType,gpuGbSize,gpuTupleNum, gpuGbKey, gpu_hashNum);
        } while(0);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbIndex));

        gbCount = 1;

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbCount,sizeof(int)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuGbCount, 0, sizeof(int)));

        do{
        	GMM_CALL(cudaReference(0, HINT_READ));
        	GMM_CALL(cudaReference(2, HINT_WRITE));
	        count_group_num<<<grid,block>>>(gpu_hashNum, HSIZE, gpuGbCount);
        } while(0);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gbCount, gpuGbCount, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_psum,HSIZE*sizeof(int)));
        scanImpl(gpu_hashNum,HSIZE,gpu_psum,pp);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbCount));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
    }

    if(gbConstant == 1)
        res->tupleNum = 1;
    else
        res->tupleNum = gbCount;

    printf("[INFO]Number of groupBy results: %d\n",res->tupleNum);

    char ** gpuResult = NULL;
    char ** result = NULL;
    
    result = (char **)malloc(sizeof(char*)*res->totalAttr);
    CHECK_POINTER(result);

#ifdef HAS_GMM
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocEx((void**)&gpuResult, sizeof(char *)* res->totalAttr, HINT_PTARRAY));
#else
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult, sizeof(char *)* res->totalAttr));
#endif

    for(int i=0; i<res->totalAttr;i++){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&result[i], res->tupleNum * res->attrSize[i]));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(result[i], 0, res->tupleNum * res->attrSize[i]));
        res->content[i] = result[i]; 
        res->dataPos[i] = GPU;
        res->attrTotalSize[i] = res->tupleNum * res->attrSize[i];
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i], &result[i], sizeof(char *), cudaMemcpyHostToDevice));
    }


    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int)*res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType, res->attrType, sizeof(int)*res->totalAttr, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int)*res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbSize, res->attrSize, sizeof(int)*res->totalAttr, cudaMemcpyHostToDevice));

    struct mathExp * gpuMathExp = NULL;
    int * cpuFunc = (int *)malloc(sizeof(int) * res->totalAttr);
    int * gpuFunc = NULL;
    int * op = (int *)malloc(sizeof(int) * res->totalAttr);
    int * gpuOp = NULL;
    int * mathExpOffset = (int *)malloc(sizeof(int) * res->totalAttr);
    int * gpuMathOffset = NULL;
    int mathExpNum = 0;

    for(int i=0;i<res->totalAttr;i++){
        mathExpOffset[i] = mathExpNum;
        cpuFunc[i] = gb->gbExp[i].func;
        op[i] = gb->gbExp[i].exp.op;
        if(gb->gbExp[i].exp.opNum == 2)
            mathExpNum += 2;
        else
            mathExpNum += 1;
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuMathExp, sizeof(struct mathExp) * mathExpNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuMathOffset, sizeof(int) * res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuMathOffset,mathExpOffset, sizeof(int) * res->totalAttr, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuFunc, sizeof(int) * res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuFunc, cpuFunc, sizeof(int) * res->totalAttr, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuOp, sizeof(int) * res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuOp, op, sizeof(int) * res->totalAttr, cudaMemcpyHostToDevice));

    for(int i=0;i<res->totalAttr;i++){
        int offset = mathExpOffset[i];
        if(gb->gbExp[i].exp.opNum == 2){
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&(gpuMathExp[offset]),(struct mathExp*)gb->gbExp[i].exp.exp,2*sizeof(struct mathExp), cudaMemcpyHostToDevice));
        }else{
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&(gpuMathExp[offset]),&(gb->gbExp[i].exp),sizeof(struct mathExp), cudaMemcpyHostToDevice));
        }
    }

    free(mathExpOffset);
    free(cpuFunc);
    free(op);

    gpuGbColNum = res->totalAttr;

    if(gbConstant !=1){
        do{
        	GMM_CALL(cudaReference(11, HINT_READ|HINT_PTARRAY|HINT_PTAWRITE));
        	GMM_CALL(cudaReference(10, HINT_READ));
        	GMM_CALL(cudaReference(0, HINT_READ|HINT_PTARRAY|HINT_PTAREAD));
        	GMM_CALL(cudaReference(3, HINT_READ));
        	GMM_CALL(cudaReference(2, HINT_READ));
        	GMM_CALL(cudaReference(5, HINT_READ));
        	GMM_CALL(cudaReference(4, HINT_READ));
        	GMM_CALL(cudaReference(7, HINT_READ));
        	GMM_CALL(cudaReference(9, HINT_READ));
	        agg_cal<<<grid,block>>>(gpuContent, gpuGbColNum, gpuFunc, gpuOp, gpuMathExp,gpuMathOffset, gpuGbType, gpuGbSize, gpuTupleNum, gpuGbKey, gpu_psum,  gpuResult);
        } while(0);
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbKey));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
    }else
		// kaibo: gpuGbKey and gpu_psum are not allocated when gbConstant == 1, so we should not reference them in ths case
        do{
        	GMM_CALL(cudaReference(11, HINT_READ|HINT_PTARRAY|HINT_PTAWRITE));
        	//GMM_CALL(cudaReference(10, HINT_READ));
        	GMM_CALL(cudaReference(0, HINT_READ|HINT_PTARRAY|HINT_PTAREAD));
        	GMM_CALL(cudaReference(3, HINT_READ));
        	GMM_CALL(cudaReference(2, HINT_READ));
        	GMM_CALL(cudaReference(5, HINT_READ));
        	GMM_CALL(cudaReference(4, HINT_READ));
        	GMM_CALL(cudaReference(7, HINT_READ));
        	//GMM_CALL(cudaReference(9, HINT_READ));
	        agg_cal_cons<<<grid,block>>>(gpuContent, gpuGbColNum, gpuFunc, gpuOp, gpuMathExp,gpuMathOffset, gpuGbType, gpuGbSize, gpuTupleNum, NULL, NULL, gpuResult);
        } while(0);

    for(int i=0; i<gb->table->totalAttr;i++){
        if(gb->table->dataPos[i]==MEM)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
    }
    free(column);
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuMathExp));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuMathOffset));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFunc));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuOp));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("GroupBy Time: %lf\n", timeE/(1000*1000));

    return res;
}
