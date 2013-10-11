/************************************************************
********
* MatrixMulti CUDA program.
************************************************************
*********/
#define BLOCK_SIZE 4
#define WIDTH (BLOCK_SIZE * 128)
#define HEIGHT (BLOCK_SIZE * 128)

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda_runtime.h>
//#include <cutil.h>
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_cuda_drvapi.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_cuda_gl.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_cuda.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_functions.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_image.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_math.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_string.h"
#include "/home/syma/etc/CUDA_5.0_SAMPLES/common/inc/helper_timer.h"

typedef struct {
int width;
int height;
float* elements;
} Matrix;
/************************************************************
************/
/* Init CUDA                                                                                                                    */
/************************************************************
************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
        int count = 0;
        int i = 0;

        cudaGetDeviceCount(&count);
        if(count == 0) {
                fprintf(stderr, "There is no device.\n");
                return false;
        }

        for(i = 0; i < count; i++) {
                cudaDeviceProp prop;
                if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                        if(prop.major >= 1) {
                                break;
                        }
                }
        }
        if(i == count) {
                fprintf(stderr, "There is no device supporting CUDA.\n");
                return false;
        }
        cudaSetDevice(i);

        printf("CUDA initialized.\n");
        return true;
}

#endif



// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
        for (int i = 0; i < size; ++i)
                data[i] = rand() / (float)RAND_MAX;

        
}

/************************************************************
************/
//Kernel

//Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int rep=0; rep<512; rep++)
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]
			* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}


static struct timeval tv0, tv1, tv2;
/************************************************************
************/
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A,d_B,d_C;
	static cudaStream_t stream = NULL;

	size_t size = A.width * A.height * sizeof(float);

	d_A.width =A.width; d_A.height = A.width;
	d_B.width = B.width; d_B.height = B.height;
	d_C.width = C.width; d_C.height = C.height;

	cudaStreamCreate(&stream);
	cudaMalloc((void**)&d_A.elements, size);
	cudaMalloc((void**)&d_B.elements, size);
	cudaMalloc((void**)&d_C.elements, size);

	gettimeofday(&tv1, NULL);
	cudaMemcpyAsync(d_A.elements,A.elements, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B.elements, B.elements, size, cudaMemcpyHostToDevice, stream);
	//cudaMemcpy(d_A.elements,A.elements, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	gettimeofday(&tv2, NULL);
        printf("P1 copying takes %ld micro seconds\n", (tv2.tv_sec - tv1.tv_sec) * 1000000L + (tv2.tv_usec - tv1.tv_usec));
	sleep(1);

	// Allocate C in device memory

	size = C.width * C.height * sizeof(float);


	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	
	cudaConfigureCall(dimGrid, dimBlock, 0, stream);
	gettimeofday(&tv1, NULL);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	cudaThreadSynchronize();
	gettimeofday(&tv2, NULL);
	printf("P1 executing kernel takes %ld micro seconds\n", (tv2.tv_sec - tv1.tv_sec) * 1000000L + (tv2.tv_usec - tv1.tv_usec));

	gettimeofday(&tv1, NULL);
	// Read C from device memory
	cudaMemcpyAsync(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost, stream);
	gettimeofday(&tv2, NULL);
	printf("P1 copying C takes %ld micro seconds\n", (tv2.tv_sec - tv1.tv_sec) * 1000000L + (tv2.tv_usec - tv1.tv_usec));

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	cudaStreamDestroy(stream);
}


/************************************************************
************/
/*MAIN                                                                                                            */
/************************************************************
************/
int main(int argc, char* argv[])
{
	if(!InitCUDA()) {
		    return 0;
	}

	gettimeofday(&tv0, NULL);
	// allocate host memory for matrices A and B
	Matrix h_A,h_B,h_C;
	h_A.width=WIDTH;
	h_A.height=HEIGHT;
	h_B.width=WIDTH;
	h_B.height=HEIGHT;
	h_C.width=WIDTH;
	h_C.height=HEIGHT;

	unsigned int size = WIDTH*HEIGHT;
	unsigned int mem_size = sizeof(float) * size;

	//h_A.elements= (float*) malloc(mem_size);
	//h_B.elements= (float*) malloc(mem_size);
	//h_C.elements= (float*) malloc(mem_size);
	cudaHostAlloc(&h_A.elements, mem_size, cudaHostAllocDefault);
	cudaHostAlloc(&h_B.elements, mem_size, cudaHostAllocDefault);
	cudaHostAlloc(&h_C.elements, mem_size, cudaHostAllocDefault);

	// set seed for rand()
	srand(2006);

	// initialize host memory
	randomInit(h_A.elements, size);
	randomInit(h_B.elements, size);

	//invoke MatMul
	MatMul(h_A,h_B,h_C);
	gettimeofday(&tv2, NULL);
        printf("P1 takes %ld micro seconds\n", (tv2.tv_sec - tv0.tv_sec) * 1000000L + (tv2.tv_usec - tv0.tv_usec));
	return 0;
}
