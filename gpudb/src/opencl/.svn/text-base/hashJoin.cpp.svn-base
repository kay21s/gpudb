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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/hashJoin.h"
#include "../include/gpuOpenclLib.h"
#include "../include/cpuOpenclLib.h"
#include "scanImpl.cpp"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)


/*
 * hashJoin implements the foreign key join between a fact table and dimension table.
 *
 * Prerequisites:
 *  1. the data to be joined can be fit into GPU device memory.
 *  2. dimension table is not compressed
 *  
 * Input:
 *  jNode: contains information about the two joined tables.
 *  pp: records statistics such as kernel execution time
 *
 * Output:
 *  A new table node
 */

struct tableNode * hashJoin(struct joinNode *jNode, struct clContext * context,struct statistic *pp){

    struct timespec start,end;
        clock_gettime(CLOCK_REALTIME,&start);

    cl_event ndrEvt;
    cl_ulong startTime, endTime;

    struct tableNode * res = NULL;

    long count = 0;
    int i;

    cl_int error = 0;

    cl_mem gpu_hashNum;
    cl_mem gpu_result;
    cl_mem  gpu_bucket, gpu_fact, gpu_dim;
    cl_mem gpu_count,  gpu_psum, gpu_resPsum;

    size_t localSize = 256;
    int blockNum = jNode->leftTable->tupleNum / localSize +1; 
    if(blockNum > 4096)
        blockNum = 4096;
    size_t globalSize = blockNum * localSize;

    size_t threadNum = globalSize;

    res = (struct tableNode*) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->totalAttr = jNode->totalAttr;
    res->tupleSize = jNode->tupleSize;
    res->attrType = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrType);
    res->attrSize = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrSize);
    res->attrIndex = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrIndex);
    res->attrTotalSize = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrTotalSize);
    res->dataPos = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->dataPos);
    res->dataFormat = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->dataFormat);
    res->content = (char **) malloc(res->totalAttr * sizeof(char *));
    CHECK_POINTER(res->content);

    for(i=0;i<jNode->leftOutputAttrNum;i++){
        int pos = jNode->leftPos[i];
        res->attrType[pos] = jNode->leftOutputAttrType[i];
        int index = jNode->leftOutputIndex[i];
        res->attrSize[pos] = jNode->leftTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    for(i=0;i<jNode->rightOutputAttrNum;i++){
        int pos = jNode->rightPos[i];
        res->attrType[pos] = jNode->rightOutputAttrType[i];
        int index = jNode->rightOutputIndex[i];
        res->attrSize[pos] = jNode->rightTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

/*
 *  build hash table on GPU
 */

    cl_mem gpu_psum1;

    int hsize = 1;
    while(hsize < jNode->rightTable->tupleNum)
        hsize *= 2;

    if(hsize ==1) hsize = 2;

    gpu_hashNum = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*hsize, NULL, &error);

    context->kernel = clCreateKernel(context->program,"cl_memset_int",0);
    clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpu_hashNum);
    int tmp = hsize;
    clSetKernelArg(context->kernel,1,sizeof(int), (void*)&tmp);
    error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->kernel += 1e-6 * (endTime - startTime);
#endif

    gpu_count = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*threadNum,NULL,&error);
    gpu_resPsum = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*threadNum,NULL,&error);

    gpu_psum = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*hsize,NULL,&error);
    gpu_bucket = clCreateBuffer(context->context,CL_MEM_READ_WRITE,2*primaryKeySize,NULL,&error);

    gpu_psum1 = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*hsize,NULL,&error);

    int dataPos = jNode->rightTable->dataPos[jNode->rightKeyIndex];

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        gpu_dim = clCreateBuffer(context->context,CL_MEM_READ_ONLY,primaryKeySize, NULL,&error);
        if (dataPos == MEM || dataPos == MMAP)
            clEnqueueWriteBuffer(context->queue,gpu_dim,CL_TRUE,0,primaryKeySize,jNode->rightTable->content[jNode->rightKeyIndex],0,0,&ndrEvt);
        else
            clEnqueueCopyBuffer(context->queue,(cl_mem)jNode->rightTable->content[jNode->rightKeyIndex],gpu_dim,0,0,primaryKeySize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_dim = (cl_mem)jNode->rightTable->content[jNode->rightKeyIndex];
    }

    context->kernel = clCreateKernel(context->program,"count_hash_num",0);
    clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_dim);
    clSetKernelArg(context->kernel,1,sizeof(long),(void*)&jNode->rightTable->tupleNum);
    clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&gpu_hashNum);
    clSetKernelArg(context->kernel,3,sizeof(int),(void*)&hsize);
    error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->kernel += 1e-6 * (endTime - startTime);
#endif

    scanImpl(gpu_hashNum,hsize,gpu_psum, context,pp);

    clEnqueueCopyBuffer(context->queue,gpu_psum,gpu_psum1,0,0,sizeof(int)*hsize,0,0,0);

    context->kernel = clCreateKernel(context->program,"build_hash_table",0);
    clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_dim);
    clSetKernelArg(context->kernel,1,sizeof(long),(void*)&jNode->rightTable->tupleNum);
    clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&gpu_psum1);
    clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void*)&gpu_bucket);
    clSetKernelArg(context->kernel,4,sizeof(int),(void*)&hsize);
    error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->kernel += 1e-6 * (endTime - startTime);
#endif

    if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
        clReleaseMemObject(gpu_dim);

    clReleaseMemObject(gpu_psum1);

/*
 *  join on GPU
 */

    cl_mem gpuFactFilter;

    dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
    int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,foreignKeySize,NULL,&error);
        if(dataPos == MEM || dataPos == MMAP)
            clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,foreignKeySize,jNode->leftTable->content[jNode->leftKeyIndex],0,0,&ndrEvt);
        else
            clEnqueueCopyBuffer(context->queue,(cl_mem)jNode->leftTable->content[jNode->leftKeyIndex],gpu_fact,0,0,foreignKeySize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_fact = (cl_mem)jNode->leftTable->content[jNode->leftKeyIndex];
    }

    gpuFactFilter = clCreateBuffer(context->context,CL_MEM_READ_WRITE,filterSize,NULL,&error);

    context->kernel = clCreateKernel(context->program,"cl_memset_int",0);
    clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuFactFilter);
    tmp = jNode->leftTable->tupleNum;
    clSetKernelArg(context->kernel,1,sizeof(int), (void*)&tmp);
    error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->kernel += 1e-6 * (endTime - startTime);
#endif

    if(format == UNCOMPRESSED){
        context->kernel = clCreateKernel(context->program,"count_join_result",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpu_hashNum);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpu_psum);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpu_bucket);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void *)&gpu_fact);
        clSetKernelArg(context->kernel,4,sizeof(long),(void *)&jNode->leftTable->tupleNum);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpu_count);
        clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void *)&gpuFactFilter);
        clSetKernelArg(context->kernel,7,sizeof(int),(void *)&hsize);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

    }else if(format == DICT){

        int dNum;
        int byteNum;
        struct dictHeader * dheader;
        cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(struct dictHeader), NULL,&error);

        if(dataPos == MEM || dataPos == MMAP){
            dheader = (struct dictHeader *) jNode->leftTable->content[jNode->leftKeyIndex];
            dNum = dheader->dictNum;
            byteNum = dheader->bitNum/8;
            clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);

        }else{
            dheader = (struct dictHeader*)clEnqueueMapBuffer(context->queue,(cl_mem)jNode->leftTable->content[jNode->leftKeyIndex],CL_TRUE,CL_MAP_READ,0,sizeof(struct dictHeader),0,0,0,0);
            dNum = dheader->dictNum;
            byteNum = dheader->bitNum/8;
            clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);
            clEnqueueUnmapMemObject(context->queue,(cl_mem)jNode->leftTable->content[jNode->leftKeyIndex],(void*)dheader,0,0,0);
        }

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        cl_mem gpuDictFilter = clCreateBuffer(context->context,CL_MEM_READ_WRITE,dNum*sizeof(int),NULL,&error);

        context->kernel = clCreateKernel(context->program,"count_join_result_dict",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpu_hashNum);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpu_psum);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpu_bucket);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void *)&gpuDictHeader);
        clSetKernelArg(context->kernel,4,sizeof(int),(void *)&dNum);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuDictFilter);
        clSetKernelArg(context->kernel,6,sizeof(int),(void *)&hsize);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        context->kernel = clCreateKernel(context->program,"transform_dict_filter_init",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpuDictFilter);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpu_fact);
        clSetKernelArg(context->kernel,2,sizeof(long),(void *)&jNode->leftTable->tupleNum);
        clSetKernelArg(context->kernel,3,sizeof(int),(void *)&dNum);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void *)&gpuFactFilter);
        clSetKernelArg(context->kernel,5,sizeof(int),(void *)&byteNum);

        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        clReleaseMemObject(gpuDictFilter);
        clReleaseMemObject(gpuDictHeader);

        context->kernel = clCreateKernel(context->program,"filter_count",0);
        clSetKernelArg(context->kernel,0,sizeof(long),(void *)&jNode->leftTable->tupleNum);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpu_count);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpuFactFilter);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

    }else if (format == RLE){

        context->kernel = clCreateKernel(context->program,"count_join_result_rle",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_hashNum);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_psum);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&gpu_bucket);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void*)&gpu_fact);
        clSetKernelArg(context->kernel,4,sizeof(long),(void*)&jNode->leftTable->tupleNum);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&gpuFactFilter);
        clSetKernelArg(context->kernel,6,sizeof(int),(void*)&hsize);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        context->kernel = clCreateKernel(context->program,"filter_count",0);
        clSetKernelArg(context->kernel,0,sizeof(long),(void *)&jNode->leftTable->tupleNum);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpu_count);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpuFactFilter);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif
    }

    int tmp1, tmp2;

    clEnqueueReadBuffer(context->queue, gpu_count, CL_TRUE, sizeof(int)*(threadNum-1), sizeof(int), &tmp1,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    scanImpl(gpu_count,threadNum,gpu_resPsum, context,pp);

    clEnqueueReadBuffer(context->queue, gpu_resPsum, CL_TRUE, sizeof(int)*(threadNum-1), sizeof(int), &tmp2,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    count = tmp1 + tmp2;
    res->tupleNum = count;
    printf("[INFO]joinNum %ld\n",count);

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        clReleaseMemObject(gpu_fact);
    }

    clReleaseMemObject(gpu_bucket);
        
    for(i=0; i<res->totalAttr; i++){
        int index, pos;
        long colSize = 0, resSize = 0;
        int leftRight = 0;

        int attrSize, attrType;
        char * table;
        int found = 0 , dataPos, format;

        if (jNode->keepInGpu[i] == 1)
            res->dataPos[i] = GPU;
        else
            res->dataPos[i] = MEM;

        for(int k=0;k<jNode->leftOutputAttrNum;k++){
            if (jNode->leftPos[k] == i){
                found = 1;
                leftRight = 0;
                pos = k;
                break;
            }
        }
        if(!found){
            for(int k=0;k<jNode->rightOutputAttrNum;k++){
                if(jNode->rightPos[k] == i){
                    found = 1;
                    leftRight = 1;
                    pos = k;
                    break;
                }
            }
        }

        if(leftRight == 0){
            index = jNode->leftOutputIndex[pos];
            dataPos = jNode->leftTable->dataPos[index];
            format = jNode->leftTable->dataFormat[index];

            table = jNode->leftTable->content[index];
            attrSize  = jNode->leftTable->attrSize[index];
            attrType  = jNode->leftTable->attrType[index];
            colSize = jNode->leftTable->attrTotalSize[index];

            resSize = res->tupleNum * attrSize;
        }else{
            index = jNode->rightOutputIndex[pos];
            dataPos = jNode->rightTable->dataPos[index];
            format = jNode->rightTable->dataFormat[index];

            table = jNode->rightTable->content[index];
            attrSize = jNode->rightTable->attrSize[index];
            attrType = jNode->rightTable->attrType[index];
            colSize = jNode->rightTable->attrTotalSize[index];

            resSize = attrSize * res->tupleNum;
            leftRight = 1;
        }


        gpu_result = clCreateBuffer(context->context,CL_MEM_READ_WRITE,resSize,NULL,&error);

        if(leftRight == 0){
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == PINNED){
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_WRITE,colSize,NULL,&error);
                    if(dataPos == MEM)
                        clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);
                    else  
                        clEnqueueCopyBuffer(context->queue,(cl_mem)table,gpu_fact,0,0,colSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else{
                    gpu_fact = (cl_mem)table;
                }

                if(attrSize == sizeof(int)){
                    context->kernel = clCreateKernel(context->program,"joinFact_int",0);
                }else{
                    context->kernel = clCreateKernel(context->program,"joinFact_other",0);
                }
                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,2,sizeof(int),(void*)&attrSize);
                clSetKernelArg(context->kernel,3,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&gpu_result);
                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif

            }else if (format == DICT){
                struct dictHeader * dheader;
                int byteNum;
                cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY,sizeof(struct dictHeader), NULL,&error);

                if(dataPos == MEM){
                    dheader = (struct dictHeader *)table;
                    byteNum = dheader->bitNum/8;
                    clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);
                }else if(dataPos == PINNED or dataPos == UVA){
                    dheader = (struct dictHeader*)clEnqueueMapBuffer(context->queue,(cl_mem)table,CL_TRUE,CL_MAP_READ,0,sizeof(struct dictHeader),0,0,0,0);
                                    byteNum = dheader->bitNum/8;
                                    clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);
                                    clEnqueueUnmapMemObject(context->queue,(cl_mem)table,(void*)dheader,0,0,0);
                }

#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->pcie += 1e-6 * (endTime - startTime);
#endif

                if(dataPos == MEM || dataPos == PINNED){
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize, NULL, &error);
                    if(dataPos == MEM)
                        clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);
                    else
                        clEnqueueCopyBuffer(context->queue,(cl_mem)table,gpu_fact,0,0,colSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else{
                    gpu_fact = (cl_mem)table;
                }

                if (attrSize == sizeof(int))
                    context->kernel = clCreateKernel(context->program,"joinFact_dict_int",0);
                else
                    context->kernel = clCreateKernel(context->program,"joinFact_dict_other",0);

                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&gpuDictHeader);
                clSetKernelArg(context->kernel,3,sizeof(int),(void*)&byteNum);
                clSetKernelArg(context->kernel,4,sizeof(int),(void*)&attrSize);
                clSetKernelArg(context->kernel,5,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void*)&gpu_result);
                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif

                clReleaseMemObject(gpuDictHeader);

            }else if (format == RLE){

                struct rleHeader *rheader;
                int dNum;
                if(dataPos == MEM){
                    rheader = (struct rleHeader*) table;
                    dNum = rheader->dictNum;
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize,NULL,&error);
                    clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else if (dataPos == PINNED){

                    rheader = (struct rleHeader*)clEnqueueMapBuffer(context->queue,(cl_mem)table,CL_TRUE,CL_MAP_READ,0,sizeof(struct rleHeader),0,0,0,0);
                                    dNum = rheader->dictNum;
                                    clEnqueueUnmapMemObject(context->queue,(cl_mem)table,(void*)rheader,0,0,0);

                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize,NULL,&error);
                    clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif

                }else if (dataPos == UVA){
                    gpu_fact = (cl_mem)table;

                    rheader = (struct rleHeader*)clEnqueueMapBuffer(context->queue,(cl_mem)table,CL_TRUE,CL_MAP_READ,0,sizeof(struct rleHeader),0,0,0,0);
                                    dNum = rheader->dictNum;
                                    clEnqueueUnmapMemObject(context->queue,(cl_mem)table,(void*)rheader,0,0,0);
                }

                cl_mem gpuRle = clCreateBuffer(context->context,CL_MEM_READ_WRITE,jNode->leftTable->tupleNum * sizeof(int), NULL, &error);;

                context->kernel = clCreateKernel(context->program,"unpack_rle",0);
                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuRle);
                clSetKernelArg(context->kernel,2,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,3,sizeof(int),(void*)&dNum);
                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif

                context->kernel = clCreateKernel(context->program,"joinFact_int",0);
                clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuRle);
                clSetKernelArg(context->kernel,2,sizeof(int), (void*)&attrSize);
                clSetKernelArg(context->kernel,3,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpu_result);

                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif


                clReleaseMemObject(gpuRle);

            }

        }else{
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == PINNED){
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize,NULL,&error);
                    if(dataPos == MEM)
                        clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);
                    else
                        clEnqueueCopyBuffer(context->queue,(cl_mem)table,gpu_fact,0,0,colSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else{
                    gpu_fact = (cl_mem)table;
                }

                if(attrSize == sizeof(int))
                    context->kernel = clCreateKernel(context->program,"joinDim_int",0);
                else
                    context->kernel = clCreateKernel(context->program,"joinDim_other",0);

                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,2,sizeof(int),(void*)&attrSize);
                clSetKernelArg(context->kernel,3,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&gpu_result);
                
                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

            }else if (format == DICT){
                struct dictHeader * dheader;
                int byteNum;
                cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(struct dictHeader), NULL, &error);

                if(dataPos == MEM){
                    dheader = (struct dictHeader *)table;
                    byteNum = dheader->bitNum/8;
                    clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);
                }else if (dataPos == PINNED || dataPos == UVA){

                    dheader = (struct dictHeader*)clEnqueueMapBuffer(context->queue,(cl_mem)table,CL_TRUE,CL_MAP_READ,0,sizeof(struct dictHeader),0,0,0,0);
                                    byteNum = dheader->bitNum/8;
                                    clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,&ndrEvt);
                                    clEnqueueUnmapMemObject(context->queue,(cl_mem)table,(void*)dheader,0,0,0);
                }

#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->pcie += 1e-6 * (endTime - startTime);
#endif

                if(dataPos == MEM || dataPos == PINNED){
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize,NULL,&error);
                    if(dataPos == MEM)
                        clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);
                    else
                        clEnqueueCopyBuffer(context->queue,(cl_mem)table,gpu_fact,0,0,colSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else{
                    gpu_fact = (cl_mem)table;
                }

                if(attrType == sizeof(int))
                    context->kernel = clCreateKernel(context->program,"joinDim_dict_int",0);
                else
                    context->kernel = clCreateKernel(context->program,"joinDim_dict_other",0);

                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&gpuDictHeader);
                clSetKernelArg(context->kernel,3,sizeof(int),(void*)&byteNum);
                clSetKernelArg(context->kernel,4,sizeof(int),(void*)&attrSize);
                clSetKernelArg(context->kernel,5,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void*)&gpu_result);

                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif
                clReleaseMemObject(gpuDictHeader);

            }else if (format == RLE){

                if(dataPos == MEM || dataPos == PINNED){
                    gpu_fact = clCreateBuffer(context->context,CL_MEM_READ_ONLY,colSize,NULL,&error);
                    if(dataPos == MEM)
                        clEnqueueWriteBuffer(context->queue,gpu_fact,CL_TRUE,0,colSize,table,0,0,&ndrEvt);
                    else
                        clEnqueueCopyBuffer(context->queue,(cl_mem)table,gpu_fact,0,0,colSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
                    clWaitForEvents(1, &ndrEvt);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                    pp->pcie += 1e-6 * (endTime - startTime);
#endif
                }else{
                    gpu_fact = (cl_mem)table;
                }

                context->kernel = clCreateKernel(context->program,"joinDim_rle",0);
                clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpu_resPsum);
                clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpu_fact);
                clSetKernelArg(context->kernel,2,sizeof(int),(void*)&attrSize);
                clSetKernelArg(context->kernel,3,sizeof(long),(void*)&jNode->leftTable->tupleNum);
                clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuFactFilter);
                clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&gpu_result);

                error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
                clWaitForEvents(1, &ndrEvt);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
                clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
                pp->kernel += 1e-6 * (endTime - startTime);
#endif
            }
        }
        
        res->attrTotalSize[i] = resSize;
        res->dataFormat[i] = UNCOMPRESSED;
        if(res->dataPos[i] == MEM){
            res->content[i] = (char *) malloc(resSize);
            CHECK_POINTER(res->content[i]);
            memset(res->content[i],0,resSize);
            clEnqueueReadBuffer(context->queue,gpu_result,CL_TRUE,0,resSize,res->content[i],0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->pcie += 1e-6 * (endTime - startTime);
#endif

            clReleaseMemObject(gpu_result);

        }else if(res->dataPos[i] == GPU){
            res->content[i] = (char *)gpu_result;
        }
        if(dataPos == MEM || dataPos == PINNED)
            clReleaseMemObject(gpu_fact);

    }

    clFinish(context->queue);
    clReleaseMemObject(gpuFactFilter);

    clReleaseMemObject(gpu_count);
    clReleaseMemObject(gpu_hashNum);
    clReleaseMemObject(gpu_psum);

    clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("HashJoin Time: %lf\n", timeE/(1000*1000));

    return res;

}
