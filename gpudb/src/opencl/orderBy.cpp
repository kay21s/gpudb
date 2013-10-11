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
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "scanImpl.cpp"
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

#define SHARED_SIZE_LIMIT 1024 
#define NTHREAD  (SHARED_SIZE_LIMIT/2)

/*
 * orderBy
 */

struct tableNode * orderBy(struct orderByNode * odNode, struct clContext *context, struct statistic *pp){
    
    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);

    cl_event ndrEvt;
    cl_ulong startTime,endTime;

    struct tableNode * res = NULL;
    size_t globalSize, localSize;

    assert(odNode->table->tupleNum < SHARED_SIZE_LIMIT);

    res = (struct tableNode *)malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->tupleNum = odNode->table->tupleNum;
    res->totalAttr = odNode->table->totalAttr;
    res->tupleSize = odNode->table->tupleSize;

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
    res->content = (char **) malloc(sizeof(char *) * res->totalAttr);
    CHECK_POINTER(res->content);

    int gpuTupleNum = odNode->table->tupleNum;
    cl_mem gpuKey, gpuContent;
    cl_mem gpuSortedKey;
    cl_mem gpuSize;
    cl_int error = 0;

    long totalSize = 0;
    long * cpuOffset = (long *)malloc(sizeof(long) * res->totalAttr);
    CHECK_POINTER(cpuOffset);
    long offset = 0;

    for(int i=0;i<res->totalAttr;i++){

        cpuOffset[i] = offset;
        res->attrType[i] = odNode->table->attrType[i];
        res->attrSize[i] = odNode->table->attrSize[i];
        res->attrTotalSize[i] = odNode->table->attrTotalSize[i];
        res->dataPos[i] = MEM;
        res->dataFormat[i] = UNCOMPRESSED;

        int size = res->attrSize[i] * res->tupleNum;

        if(size %4 !=0){
            size += (4 - size %4);
        }

        offset += size;
        totalSize += size;
    }

    gpuContent = clCreateBuffer(context->context,CL_MEM_READ_ONLY, totalSize, NULL, 0);

    for(int i=0;i<res->totalAttr;i++){

        int size = res->attrSize[i] * res->tupleNum;

        if(odNode->table->dataPos[i] == MEM){
            error = clEnqueueWriteBuffer(context->queue, gpuContent, CL_TRUE, cpuOffset[i], size, odNode->table->content[i],0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->pcie += 1e-6 * (endTime - startTime);
#endif
        }else if (odNode->table->dataPos[i] == GPU){
            error = clEnqueueCopyBuffer(context->queue,(cl_mem)odNode->table->content[i],gpuContent,0,cpuOffset[i],size,0,0,0);
        }

    }

    cl_mem gpuOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*res->totalAttr,NULL,0);
    error = clEnqueueWriteBuffer(context->queue, gpuOffset, CL_TRUE, 0, sizeof(long)*res->totalAttr, cpuOffset,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    int newNum = 1;

    while(newNum<gpuTupleNum){
        newNum *=2;
    }

    int dir;
    if(odNode->orderBySeq[0] == ASC)
        dir = 1;
    else
        dir = 0;

    int index = odNode->orderByIndex[0];
    int type = odNode->table->attrType[index];

    cl_mem gpuPos = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*newNum, NULL,0);
    gpuSize = clCreateBuffer(context->context,CL_MEM_READ_ONLY, res->totalAttr * sizeof(int), NULL, 0);
    clEnqueueWriteBuffer(context->queue, gpuSize, CL_TRUE, 0, sizeof(int)*res->totalAttr, res->attrSize,0,0,&ndrEvt);
    cl_mem gpuResult = clCreateBuffer(context->context,CL_MEM_READ_WRITE, totalSize, NULL,0);
    
    long * resOffset = (long *) malloc(sizeof(long) * res->totalAttr);
    CHECK_POINTER(resOffset);
    offset = 0;
    totalSize = 0;
    for(int i=0; i<res->totalAttr;i++){
        int size = res->attrSize[i] * res->tupleNum;
        if(size %4 != 0){
            size += 4 - (size % 4);
        }

        resOffset[i] = offset;
        offset += size;
        totalSize += size;
    }
    
    cl_mem gpuResOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*res->totalAttr, NULL,0);
    clEnqueueWriteBuffer(context->queue, gpuResOffset, CL_TRUE, 0 ,sizeof(long)*res->totalAttr, resOffset, 0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif
    if(type == INT){

        gpuKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int) * newNum, NULL, 0);
        gpuSortedKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int) * newNum, NULL, 0);
        
        localSize = 128;
        globalSize = 8*localSize;
        context->kernel = clCreateKernel(context->program,"set_key_int",0);
        error = clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuKey);
        error = clSetKernelArg(context->kernel,1,sizeof(int), (void *)&newNum);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif

        clEnqueueWriteBuffer(context->queue, gpuKey, CL_TRUE, 0, sizeof(int)*gpuTupleNum, odNode->table->content[index],0,0,&ndrEvt);
        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
        #endif
        
        context->kernel = clCreateKernel(context->program,"sort_key_int",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
        clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&gpuSortedKey);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuPos);
        clSetKernelArg(context->kernel,4,sizeof(int), (void*)&dir);
        clSetKernelArg(context->kernel,5,SHARED_SIZE_LIMIT*sizeof(int), NULL);
        clSetKernelArg(context->kernel,6,SHARED_SIZE_LIMIT*sizeof(int), NULL);

        localSize = newNum/2;
        globalSize = localSize;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif

    }else if (type == FLOAT){

        gpuKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(float) * newNum, NULL, 0);
        gpuSortedKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(float) * newNum, NULL, 0);
        
        localSize = 128;
        globalSize = 8*localSize;
        context->kernel = clCreateKernel(context->program,"set_key_float",0);
        error = clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuKey);
        error = clSetKernelArg(context->kernel,1,sizeof(int), (void *)&newNum);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif

        clEnqueueWriteBuffer(context->queue, gpuKey, CL_TRUE, 0, sizeof(float)*gpuTupleNum, odNode->table->content[index],0,0,&ndrEvt);
        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
        #endif
        
        context->kernel = clCreateKernel(context->program,"sort_key_float",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
        clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&gpuSortedKey);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuPos);
        clSetKernelArg(context->kernel,4,sizeof(int), (void*)&dir);
        clSetKernelArg(context->kernel,5,SHARED_SIZE_LIMIT*sizeof(float), NULL);
        clSetKernelArg(context->kernel,6,SHARED_SIZE_LIMIT*sizeof(int), NULL);

        localSize = newNum/2;
        globalSize = localSize;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif


    }else if (type == STRING){
        int keySize = odNode->table->attrSize[index];

        gpuKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize * newNum, NULL, 0);
        gpuSortedKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize * newNum, NULL, 0);
        
        localSize = 128;
        globalSize = 8*localSize;
        context->kernel = clCreateKernel(context->program,"set_key_string",0);
        int tmp = newNum * keySize;
        error = clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuKey);
        error = clSetKernelArg(context->kernel,1,sizeof(int), (void *)&tmp);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif

        clEnqueueWriteBuffer(context->queue, gpuKey, CL_TRUE, 0, keySize*gpuTupleNum, odNode->table->content[index],0,0,&ndrEvt);
        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
        #endif
        
        context->kernel = clCreateKernel(context->program,"sort_key_string",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
        clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
        clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuSortedKey);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
        clSetKernelArg(context->kernel,5,sizeof(int), (void*)&dir);
        clSetKernelArg(context->kernel,6,SHARED_SIZE_LIMIT*keySize, NULL);
        clSetKernelArg(context->kernel,7,SHARED_SIZE_LIMIT*sizeof(int), NULL);

        localSize = newNum/2;
        globalSize = localSize;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif

    }

    if (odNode->orderByNum == 2){
        int keySize = odNode->table->attrSize[index];
        int secIndex = odNode->orderByIndex[1];
        int keySize2 = odNode->table->attrSize[secIndex];
        int secType = odNode->table->attrType[secIndex];
        cl_mem keyNum , keyCount, keyPsum;

        keyNum = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int), NULL, 0);

        if(type == INT){
            context->kernel = clCreateKernel(context->program,"count_unique_keys_int",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyNum);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif
        }else if (type == FLOAT){

            context->kernel = clCreateKernel(context->program,"count_unique_keys_float",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyNum);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif
        }else if (type == STRING){
            context->kernel = clCreateKernel(context->program,"count_unique_keys_string",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
            clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&keyNum);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif
        }

        int cpuKeyNum;
        clEnqueueReadBuffer(context->queue,keyNum, CL_TRUE, 0, sizeof(int), &cpuKeyNum,0,0,&ndrEvt);

        keyCount = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int)*cpuKeyNum, NULL,0);
        keyPsum = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int)*cpuKeyNum, NULL,0);

        if(type == INT){
            context->kernel = clCreateKernel(context->program,"count_key_num_int",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyCount);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif
        }else if (type == FLOAT){
            context->kernel = clCreateKernel(context->program,"count_key_num_float",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyCount);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

        }else if (type == STRING){
            context->kernel = clCreateKernel(context->program,"count_key_num_string",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuSortedKey);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
            clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&keyCount);
            localSize = 1;
            globalSize = 1;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif
        }
        scanImpl(keyCount, cpuKeyNum, keyPsum, context,pp);

        cl_mem gpuPos2, gpuKey2;
        gpuPos2 = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int)*newNum, NULL,0);
        gpuKey2 = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize2*newNum, NULL,0);

        if(secType == INT){

            context->kernel = clCreateKernel(context->program,"gather_col_int",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&odNode->table->content[secIndex]);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&newNum);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuKey2);
            localSize = 128;
            globalSize = 8*128;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

            context->kernel = clCreateKernel(context->program,"sec_sort_key_int",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey2);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&keyPsum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyCount);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuPos2);
            localSize = 1;
            globalSize = cpuKeyNum;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

        }else if (secType == FLOAT){

            context->kernel = clCreateKernel(context->program,"gather_col_float",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&odNode->table->content[secIndex]);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&newNum);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuKey2);
            localSize = 128;
            globalSize = 8*128;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

            context->kernel = clCreateKernel(context->program,"sec_sort_key_float",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey2);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&keyPsum);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyCount);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuPos2);
            localSize = 1;
            globalSize = cpuKeyNum;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif


        }else if (secType == STRING){

            context->kernel = clCreateKernel(context->program,"gather_col_string",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&odNode->table->content[secIndex]);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&newNum);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&keySize2);
            clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuKey2);
            localSize = 128;
            globalSize = 8*128;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

            context->kernel = clCreateKernel(context->program,"sec_sort_key_string",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey2);
            clSetKernelArg(context->kernel,1,sizeof(int), (void*)&keySize);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&keyPsum);
            clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&keyCount);
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&gpuTupleNum);
            clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuPos);
            clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuPos2);
            localSize = 1;
            globalSize = cpuKeyNum;
            error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
            #ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->kernel += 1e-6 * (endTime - startTime);
            #endif

        }

        context->kernel = clCreateKernel(context->program,"gather_result",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpuPos2);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuContent);
        clSetKernelArg(context->kernel,2,sizeof(int),(void*)&newNum);
        clSetKernelArg(context->kernel,3,sizeof(int),(void*)&gpuTupleNum);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuSize);
        clSetKernelArg(context->kernel,5,sizeof(int),(void*)&res->totalAttr);
        clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuResult);
        clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void*)&gpuOffset);
        clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void*)&gpuResOffset);

        localSize = 128;
        globalSize = 8 * localSize;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif


        clReleaseMemObject(keyCount);
        clReleaseMemObject(keyNum);
        clReleaseMemObject(gpuPos2);
        clReleaseMemObject(gpuKey2);
    }else{
 
        context->kernel = clCreateKernel(context->program,"gather_result",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpuPos);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuContent);
        clSetKernelArg(context->kernel,2,sizeof(int),(void*)&newNum);
        clSetKernelArg(context->kernel,3,sizeof(int),(void*)&gpuTupleNum);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuSize);
        clSetKernelArg(context->kernel,5,sizeof(int),(void*)&res->totalAttr);
        clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuResult);
        clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void*)&gpuOffset);
        clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void*)&gpuResOffset);

        localSize = 128;
        globalSize = 8 * localSize;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

        #ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
        #endif
    }

   

    for(int i=0; i<res->totalAttr;i++){
        int size = res->attrSize[i] * gpuTupleNum;
        res->content[i] = (char *) malloc( size);
        CHECK_POINTER(res->content[i]);
        memset(res->content[i],0, size);
        clEnqueueReadBuffer(context->queue,gpuResult, CL_TRUE, resOffset[i], size, res->content[i],0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif
    }

    free(resOffset);
    clFinish(context->queue);
    clReleaseMemObject(gpuKey);
    clReleaseMemObject(gpuContent);
    clReleaseMemObject(gpuResult);
    clReleaseMemObject(gpuSize);
    clReleaseMemObject(gpuPos);
    clReleaseMemObject(gpuOffset);
    clReleaseMemObject(gpuResOffset);

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("orderBy Time: %lf\n", timeE/(1000*1000));

    return res;
}
