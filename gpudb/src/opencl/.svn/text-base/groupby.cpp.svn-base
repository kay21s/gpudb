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
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"
#include "../include/cpuOpenclLib.h"
#include "scanImpl.cpp"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)


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


struct tableNode * groupBy(struct groupByNode * gb, struct clContext * context, struct statistic * pp){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);

    cl_event ndrEvt;
    cl_ulong startTime,endTime;

    struct tableNode * res = NULL;
    long gpuTupleNum;
    int gpuGbColNum;
    cl_mem gpuGbIndex;
    cl_mem gpuGbType, gpuGbSize;

    cl_mem gpuGbKey;
    cl_mem gpuContent;

    int gbCount;                // the number of groups
    int gbConstant = 0;         // whether group by constant

    cl_int error = 0;

    res = (struct tableNode *) malloc(sizeof(struct tableNode));
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

    size_t localSize = 128;
    size_t globalSize = 1024*128;

    int blockNum = gb->table->tupleNum / localSize + 1; 

    if(blockNum < 1024)
        globalSize = blockNum * 128;


    cl_mem gpu_hashNum;
    cl_mem gpu_psum;
    cl_mem gpuGbCount;

    long * cpuOffset = (long *)malloc(sizeof(long) * gb->table->totalAttr);
    CHECK_POINTER(cpuOffset);
    long offset = 0;
    long totalSize = 0;

    for(int i=0;i<gb->table->totalAttr;i++){

        int attrSize = gb->table->attrSize[i];
        int size = attrSize * gb->table->tupleNum;

        cpuOffset[i] = offset;

        /*align each column*/

        if(size % 4 !=0){
            size += 4 - (size%4);
        }

        offset += size;
        totalSize += size;
    }

    gpuContent = clCreateBuffer(context->context,CL_MEM_READ_ONLY, totalSize,NULL,&error);

    for(int i=0;i<gb->table->totalAttr;i++){

        int attrSize = gb->table->attrSize[i];
        int size = attrSize * gb->table->tupleNum;

        if(gb->table->dataPos[i]==MEM){
            error = clEnqueueWriteBuffer(context->queue, gpuContent, CL_TRUE, cpuOffset[i], size, gb->table->content[i],0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->pcie += 1e-6 * (endTime - startTime);
#endif
        }else
            error = clEnqueueCopyBuffer(context->queue,(cl_mem)gb->table->content[i],gpuContent,0, cpuOffset[i],size,0,0,0);

    }

    cl_mem gpuOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*gb->table->totalAttr,NULL,&error);
    clEnqueueWriteBuffer(context->queue,gpuOffset,CL_TRUE,0,sizeof(long)*gb->table->totalAttr,cpuOffset,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    if(gbConstant != 1){

        gpuGbType = clCreateBuffer(context->context,CL_MEM_READ_ONLY,sizeof(int)*gb->groupByColNum,NULL,&error);
        clEnqueueWriteBuffer(context->queue,gpuGbType,CL_TRUE,0,sizeof(int)*gb->groupByColNum,gb->groupByType,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        gpuGbSize = clCreateBuffer(context->context,CL_MEM_READ_ONLY,sizeof(int)*gb->groupByColNum,NULL,&error);
        clEnqueueWriteBuffer(context->queue,gpuGbSize,CL_TRUE,0,sizeof(int)*gb->groupByColNum,gb->groupBySize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        gpuGbKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*gb->table->tupleNum,NULL,&error);

        gpuGbIndex = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(int)*gb->groupByColNum,NULL,&error);
        clEnqueueWriteBuffer(context->queue,gpuGbIndex,CL_TRUE,0,sizeof(int)*gb->groupByColNum,gb->groupByIndex,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        gpu_hashNum = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int)*HSIZE,NULL,&error);

        context->kernel = clCreateKernel(context->program,"cl_memset_int",0);

        int tmp = HSIZE;
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpu_hashNum);
        clSetKernelArg(context->kernel,1,sizeof(int), (void*)&tmp);

        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        context->kernel = clCreateKernel(context->program, "build_groupby_key",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpuContent);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpuOffset);
        clSetKernelArg(context->kernel,2,sizeof(int),(void *)&gpuGbColNum);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void *)&gpuGbIndex);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void *)&gpuGbType);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuGbSize);
        clSetKernelArg(context->kernel,6,sizeof(long),(void *)&gpuTupleNum);
        clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void *)&gpuGbKey);
        clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void *)&gpu_hashNum);

        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        clReleaseMemObject(gpuGbType);
        clReleaseMemObject(gpuGbSize);
        clReleaseMemObject(gpuGbIndex);

        gbCount = 1;

        tmp = 0;
        gpuGbCount = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,&error);
        clEnqueueWriteBuffer(context->queue,gpuGbCount,CL_TRUE,0,sizeof(int),&tmp,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        int hsize = HSIZE;
        context->kernel = clCreateKernel(context->program, "count_group_num",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpu_hashNum);
        clSetKernelArg(context->kernel,1,sizeof(int),(void *)&hsize);
        clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpuGbCount);
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif

        clEnqueueReadBuffer(context->queue, gpuGbCount, CL_TRUE, 0, sizeof(int), &gbCount,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        gpu_psum = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int)*HSIZE,NULL,&error);

        scanImpl(gpu_hashNum,HSIZE,gpu_psum,context,pp);

        clReleaseMemObject(gpuGbCount);
        clReleaseMemObject(gpu_hashNum);
    }

    if(gbConstant == 1)
        res->tupleNum = 1;
    else
        res->tupleNum = gbCount;

    printf("groupBy num %ld\n",res->tupleNum);

    gpuGbType = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(int)*res->totalAttr, NULL, &error);
    clEnqueueWriteBuffer(context->queue,gpuGbType,CL_TRUE,0,sizeof(int)*res->totalAttr,res->attrType,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    gpuGbSize = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(int)*res->totalAttr, NULL, &error);
    clEnqueueWriteBuffer(context->queue,gpuGbSize,CL_TRUE,0,sizeof(int)*res->totalAttr,res->attrSize,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    /*
     * @gpuGbExp is the mathExp in each groupBy expression
     * @mathexp stores the math exp for for the group expression that has two operands
     * The reason that we need two variables instead of one is that OpenCL doesn't support pointer to pointer
     * */

    cl_mem gpuGbExp = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(struct mathExp)*res->totalAttr, NULL, &error);
    cl_mem mathexp = clCreateBuffer(context->context, CL_MEM_READ_ONLY, 2*sizeof(struct mathExp)*res->totalAttr, NULL, &error);

    struct mathExp tmpExp[2];
    int * cpuFunc = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(cpuFunc);

    offset = 0;
    for(int i=0;i<res->totalAttr;i++){

        error = clEnqueueWriteBuffer(context->queue, gpuGbExp, CL_TRUE, offset, sizeof(struct mathExp), &(gb->gbExp[i].exp),0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->pcie += 1e-6 * (endTime - startTime);
#endif

        offset += sizeof(struct mathExp);

        cpuFunc[i] = gb->gbExp[i].func;

        if(gb->gbExp[i].exp.opNum == 2){
            struct mathExp * tmpMath = (struct mathExp *) (gb->gbExp[i].exp.exp);
            tmpExp[0].op = tmpMath[0].op;
            tmpExp[0].opNum = tmpMath[0].opNum;
            tmpExp[0].opType = tmpMath[0].opType;
            tmpExp[0].opValue = tmpMath[0].opValue;

            tmpExp[1].op = tmpMath[1].op;
            tmpExp[1].opNum = tmpMath[1].opNum;
            tmpExp[1].opType = tmpMath[1].opType;
            tmpExp[1].opValue = tmpMath[1].opValue;
            clEnqueueWriteBuffer(context->queue, mathexp, CL_TRUE, 2*i*sizeof(struct mathExp),2*sizeof(struct mathExp),tmpExp,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
            clWaitForEvents(1, &ndrEvt);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
            clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
            pp->pcie += 1e-6 * (endTime - startTime);
#endif
        }
    }

    cl_mem gpuFunc = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(int)*res->totalAttr, NULL, &error);
    clEnqueueWriteBuffer(context->queue,gpuFunc,CL_TRUE,0,sizeof(int)*res->totalAttr,cpuFunc,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    long *resOffset = (long *)malloc(sizeof(long)*res->totalAttr);
    CHECK_POINTER(resOffset);
    
    offset = 0;
    totalSize = 0;
    for(int i=0;i<res->totalAttr;i++){
        
        /*
         * align the output of each column on the boundary of 4
         */

        int size = res->attrSize[i] * res->tupleNum;
        if(size %4 != 0){
            size += 4- (size %4);
        }

        resOffset[i] = offset;
        offset += size; 
        totalSize += size;
    }

    cl_mem gpuResult = clCreateBuffer(context->context,CL_MEM_READ_WRITE, totalSize, NULL, &error);
    cl_mem gpuResOffset = clCreateBuffer(context->context, CL_MEM_READ_ONLY,sizeof(long)*res->totalAttr, NULL,&error);
    clEnqueueWriteBuffer(context->queue,gpuResOffset,CL_TRUE,0,sizeof(long)*res->totalAttr,resOffset,0,0,&ndrEvt);

#ifdef OPENCL_PROFILE
    clWaitForEvents(1, &ndrEvt);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
    clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
    pp->pcie += 1e-6 * (endTime - startTime);
#endif

    gpuGbColNum = res->totalAttr;

    if(gbConstant !=1){
        context->kernel = clCreateKernel(context->program,"agg_cal",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuOffset);
        clSetKernelArg(context->kernel,2,sizeof(int), (void*)&gpuGbColNum);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuGbExp);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&mathexp);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuGbType);
        clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuGbSize);
        clSetKernelArg(context->kernel,7,sizeof(long), (void*)&gpuTupleNum);
        clSetKernelArg(context->kernel,8,sizeof(cl_mem), (void*)&gpuGbKey);
        clSetKernelArg(context->kernel,9,sizeof(cl_mem), (void*)&gpu_psum);
        clSetKernelArg(context->kernel,10,sizeof(cl_mem), (void*)&gpuResult);
        clSetKernelArg(context->kernel,11,sizeof(cl_mem), (void*)&gpuResOffset);
        clSetKernelArg(context->kernel,12,sizeof(cl_mem), (void*)&gpuFunc);

        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif
        
        clReleaseMemObject(gpuGbKey);
        clReleaseMemObject(gpu_psum);
    }else{
        context->kernel = clCreateKernel(context->program,"agg_cal_cons",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuOffset);
        clSetKernelArg(context->kernel,2,sizeof(int), (void*)&gpuGbColNum);
        clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuGbExp);
        clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&mathexp);
        clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuGbType);
        clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuGbSize);
        clSetKernelArg(context->kernel,7,sizeof(long), (void*)&gpuTupleNum);
        clSetKernelArg(context->kernel,8,sizeof(cl_mem), (void*)&gpuResult);
        clSetKernelArg(context->kernel,9,sizeof(cl_mem), (void*)&gpuResOffset);
        clSetKernelArg(context->kernel,10,sizeof(cl_mem), (void*)&gpuFunc);

        globalSize = localSize * 4;
        error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,&ndrEvt);
#ifdef OPENCL_PROFILE
        clWaitForEvents(1, &ndrEvt);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
        clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
        pp->kernel += 1e-6 * (endTime - startTime);
#endif
    }

    for(int i=0; i<res->totalAttr;i++){
        res->content[i] = (char *)clCreateBuffer(context->context,CL_MEM_READ_WRITE, res->attrSize[i]*res->tupleNum, NULL, &error); 
        res->dataPos[i] = GPU;
        res->attrTotalSize[i] = res->tupleNum * res->attrSize[i];
        clEnqueueCopyBuffer(context->queue, gpuResult, (cl_mem)res->content[i], resOffset[i],0, res->attrSize[i] * res->tupleNum, 0,0,0);
    }

    free(resOffset);
    free(cpuOffset);

    clFinish(context->queue);
    clReleaseMemObject(gpuContent);
    clReleaseMemObject(gpuResult);
    clReleaseMemObject(gpuOffset);
    clReleaseMemObject(gpuResOffset);
    clReleaseMemObject(gpuGbExp);
    clReleaseMemObject(gpuFunc);

    clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("GroupBy Time: %lf\n", timeE/(1000*1000));

    return res;
}
