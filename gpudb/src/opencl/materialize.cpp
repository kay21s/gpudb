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
#include "../include/schema.h"
#include "../include/gpuOpenclLib.h"

void * materializeCol(struct materializeNode * mn, struct clContext * context, struct statistic * pp){

	struct timespec start,end;
        clock_gettime(CLOCK_REALTIME,&start);

	cl_event ndrEvt;
	cl_ulong startTime, endTime;

	struct tableNode *tn = mn->table;
	char * res;
	cl_mem gpuResult;
	cl_mem gpuAttrSize;

	long totalSize = tn->tupleNum * tn->tupleSize;

	cl_int error = 0;

	cl_mem gpuContent = clCreateBuffer(context->context, CL_MEM_READ_ONLY, totalSize, NULL, &error);
	gpuResult = clCreateBuffer(context->context, CL_MEM_READ_WRITE, totalSize, NULL, &error);
	gpuAttrSize = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(int)*tn->totalAttr,NULL,&error);
	clEnqueueWriteBuffer(context->queue,gpuAttrSize,CL_TRUE,0,sizeof(int)*tn->totalAttr,tn->attrSize,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

	res = (char *) malloc(totalSize);

	long offset = 0;
	long *colOffset = (long*)malloc(sizeof(long)*tn->totalAttr);

	for(int i=0;i<tn->totalAttr;i++){
		colOffset[i] = offset;
		int size = tn->tupleNum * tn->attrSize[i]; 

		if(tn->dataPos[i] == MEM){
			clEnqueueWriteBuffer(context->queue,gpuContent,CL_TRUE,offset,size,tn->content[i],0,0,&ndrEvt);

			clWaitForEvents(1, &ndrEvt);
			clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
			clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
			pp->pcie += 1e-6 * (endTime - startTime);
		}else
			clEnqueueCopyBuffer(context->queue,(cl_mem)tn->content[i],gpuContent,0,offset,size,0,0,0);
			
		offset += size;
	}

	cl_mem gpuColOffset = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(long)*tn->totalAttr,NULL,&error);
	clEnqueueWriteBuffer(context->queue,gpuColOffset,CL_TRUE,0,sizeof(long)*tn->totalAttr,colOffset,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

	size_t globalSize = 512;
	size_t localSize = 128;

	context->kernel = clCreateKernel(context->program,"materialize",0);
	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuColOffset);
	clSetKernelArg(context->kernel,2,sizeof(int), (void*)&tn->totalAttr);
	clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuAttrSize);
	clSetKernelArg(context->kernel,4,sizeof(long), (void*)&tn->tupleNum);
	clSetKernelArg(context->kernel,5,sizeof(int), (void*)&tn->tupleSize);
	clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuResult);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	clEnqueueReadBuffer(context->queue,gpuResult,CL_TRUE,0,totalSize,res,0,0,&ndrEvt);
	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

	free(colOffset);

	clFinish(context->queue);

	clReleaseMemObject(gpuColOffset);
	clReleaseMemObject(gpuContent);
	clReleaseMemObject(gpuAttrSize);
	clReleaseMemObject(gpuResult);


	clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("Materialization Time: %lf\n", timeE/(1000*1000));
	return res;
}
