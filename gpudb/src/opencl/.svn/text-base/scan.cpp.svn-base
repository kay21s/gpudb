#ifndef _PRESCAN_CPP
#define _PRESCAN_CPP_

#include <assert.h>
#include <CL/cl.h> 
#include <math.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"

static inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

static inline int 
floorPow2(int n)
{
    int exp;
    frexp((int)n, &exp);
    return 1 << (exp - 1);
}

#define BLOCK_SIZE 256
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

static cl_mem * g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static int g_numLevelsAllocated = 0;

static int max(int a, int b){
	if(a>b)
		return a;
	else 
		return b;
}

static void preallocBlockSums(unsigned int maxNumElements, struct clContext *context)
{
    assert(g_numEltsAllocated == 0); 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; 
    unsigned int numElts = maxNumElements;

    int level = 0;
    cl_int error = 0;

    do
    {       
        unsigned int numBlocks = max(1, (int)ceil((int)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (cl_mem *) malloc(level * sizeof(cl_mem));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {
        unsigned int numBlocks = max(1, (int)ceil((int)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            g_scanBlockSums[level++] = clCreateBuffer(context->context,CL_MEM_READ_WRITE, numBlocks*sizeof(int), NULL, &error);
        }
        numElts = numBlocks;
    } while (numElts > 1);

}

static void deallocBlockSums()
{
    for (int i = 0; i < g_numLevelsAllocated; i++)
    {
        clReleaseMemObject(g_scanBlockSums[i]);
    }

    
    free(g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


static void prescanArrayRecursive(cl_mem outArray, cl_mem inArray, int numElements, int level, int same, struct clContext *context, struct statistic *pp)
{

    unsigned int blockSize = BLOCK_SIZE; 
    unsigned int numBlocks = max(1, (int)ceil((int)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
    }

    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

    size_t localSize = numThreads;
    size_t globalSize = max(1, numBlocks-np2LastBlock) * localSize; 

    int tmp = 0;
    if (numBlocks > 1)
    {

        context->kernel = clCreateKernel(context->program,"prescan",0);

        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);
	if(same == 0)
        	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&inArray);
	else{
		cl_mem tmpP = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,0);
        	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void *)&tmpP);
	}
		
        clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&g_scanBlockSums[level]);

	tmp = numThreads * 2;
        clSetKernelArg(context->kernel,3,sizeof(int), (void*)&(tmp));
	tmp = 0;
        clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
        clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(tmp));
	tmp = 1;
        clSetKernelArg(context->kernel,6,sizeof(int), (void*)&(tmp));
	tmp = 0;
        clSetKernelArg(context->kernel,7,sizeof(int), (void*)&(tmp));
        clSetKernelArg(context->kernel,8,sizeof(int), (void*)&same);
        clSetKernelArg(context->kernel,9,sharedMemSize, NULL);

        clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

        if (np2LastBlock)
        {
            context->kernel = clCreateKernel(context->program,"prescan",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);

	    if(same == 0)
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&inArray);
	    else{
		cl_mem tmpP = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,0);
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&tmpP);
	    }

            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&g_scanBlockSums[level]);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&numEltsLastBlock);

	    tmp = numBlocks -1 ;
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
	    tmp = numElements - numEltsLastBlock;
            clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(tmp));
	    tmp = 1;
            clSetKernelArg(context->kernel,6,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,7,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,8,sizeof(int), (void*)&same);
            clSetKernelArg(context->kernel,9,sharedMemLastBlock, NULL);

            globalSize = localSize = numThreadsLastBlock;
            clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

        }

        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1,1,context,pp);

        context->kernel = clCreateKernel(context->program,"uniformAdd",0);
        clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);
        clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&g_scanBlockSums[level]);
	tmp = numElements - numEltsLastBlock;
        clSetKernelArg(context->kernel,2,sizeof(int), (void*)&(tmp));
        tmp = 0;
        clSetKernelArg(context->kernel,3,sizeof(int), (void*)&(tmp));
        clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
        clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(numElements));

        localSize = numThreads;
        globalSize = max(1, numBlocks-np2LastBlock) * localSize; 
        clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

        if (np2LastBlock)
        {
            context->kernel = clCreateKernel(context->program,"uniformAdd",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);
            clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&g_scanBlockSums[level]);
            clSetKernelArg(context->kernel,2,sizeof(int), (void*)&numEltsLastBlock);
            tmp = numBlocks -1;
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&(tmp));
            tmp = numElements - numEltsLastBlock;
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(numElements));

            globalSize = localSize = numThreadsLastBlock;
            clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
        }
    }
    else if (isPowerOfTwo(numElements))
    {
            context->kernel = clCreateKernel(context->program,"prescan",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);

	    if(same ==0)
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&inArray);
	    else{
		cl_mem tmpP = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,0);
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&tmpP);
	    }

	    cl_mem tmpP = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,0);
            clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&tmpP);
	    tmp = numThreads * 2;
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&(tmp));
	    tmp = 0;
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,6,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,7,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,8,sizeof(int), (void*)&same);
            clSetKernelArg(context->kernel,9,sharedMemSize, NULL);

            localSize = numThreads;
            globalSize = max(1, numBlocks-np2LastBlock) * localSize; 
            clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

    }
    else
    {
            context->kernel = clCreateKernel(context->program,"prescan",0);
            clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&outArray);

	    if(same ==0)
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&inArray);
	    else{
		cl_mem tmpP = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int),NULL,0);
            	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&tmpP);
	    }

            clSetKernelArg(context->kernel,2,sizeof(cl_mem), NULL);
            clSetKernelArg(context->kernel,3,sizeof(int), (void*)&numElements);
	    tmp = 0;
            clSetKernelArg(context->kernel,4,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,5,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,6,sizeof(int), (void*)&(tmp));
	    tmp = 1;
            clSetKernelArg(context->kernel,7,sizeof(int), (void*)&(tmp));
            clSetKernelArg(context->kernel,8,sizeof(int), (void*)&same);
            clSetKernelArg(context->kernel,9,sharedMemSize, NULL);

            localSize = numThreads;
            globalSize = max(1, numBlocks-np2LastBlock) * localSize; 
            clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

    }
}

static void prescanArray(cl_mem outArray, cl_mem inArray, int numElements, struct clContext * context, struct statistic *pp)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0, 0,context,pp);
}


#endif 
