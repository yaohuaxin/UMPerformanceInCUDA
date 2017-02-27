/*****************************************************
 * This file tests cuda memory management APIs.
 *****************************************************/
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define initializingMode_ 0
#define dataBlockSize_ 16*1024*1024
#define totalLoop_ 1024*2

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	cudaError_t result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		printf("Error: %s.\n", cudaGetErrorString(result));             \
		exit(1);														\
	} }

/*
 * Initialzing and computing kernels
 */
__global__ void vecInit(float *A, float value) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("In GPU: %f.\n", value);
	A[i] = value;

}

__global__ void vecAdd(float* A, float* B, float* C) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
	//printf("In GPU:\n");
}

__global__ void vecMultiply(float* A, float* B, float* C) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] * B[i];
}

__global__ void vecMultiplyAndAdd(float* A, float* B, float* C, float* D) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	D[i] = (A[i] + B[i]) * C[i];
}

double currentTimeCPUSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void test_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
	cudaDeviceGetAttribute(value, attr, device);
}

void initialData(float *A, unsigned int n, float data, int mode = 0) {
	//Using GPU to initialize the data, so that no need to copy memory from GPU to CPU
	//printf("Initialize the data.\n");
	if (mode == 0) {
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((n+threadsPerBlock.x-1) / threadsPerBlock.x);
		//printf("Threads per block: %d, Blocks: %d.\n", threadsPerBlock.x, numBlocks.x);
		vecInit<<<numBlocks, threadsPerBlock>>>(A, data);
		cudaDeviceSynchronize();
	} else {
		unsigned int i;
		for (i = 0; i < n; i++) {
			A[i] = data;
		}
	}
}

void profileAddtionBasement(unsigned int nElement, unsigned int totalLoop) {
	unsigned int nBytes = nElement * sizeof(float);

	if (nBytes < 1024) {
		printf("Allocate nbytes is %d B.\n", nBytes);
	} else if (nBytes >= 1024 && nBytes < 1024*1024) {
		printf("Allocate nbytes is %d KB.\n", nBytes/1024);
	} else {
		printf("Allocate nbytes is %d MB.\n", nBytes/(1024*1024));
	}

	//profile device memory as the basement
	float *memoryOnDevice_A, *memoryOnDevice_B, *memoryOnDevice_C;
	CHECK_CUDA_RESULT(cudaMalloc(&memoryOnDevice_A, nBytes));
	CHECK_CUDA_RESULT(cudaMalloc(&memoryOnDevice_B, nBytes));
	CHECK_CUDA_RESULT(cudaMalloc(&memoryOnDevice_C, nBytes));

	printf("===== inital data begins...\n");
	int mode = initializingMode_;
	double iStart = currentTimeCPUSecond();
	{
		initialData(memoryOnDevice_A, nElement, 2.0f, mode);
		initialData(memoryOnDevice_B, nElement, 2.0f, mode);
		initialData(memoryOnDevice_C, nElement, 0.0f, mode);
	}
	double iStop = currentTimeCPUSecond();
	if (mode == 0) {
		printf("==== GPU mode: time for initializing the data: %f.\n", (iStop - iStart)*totalLoop);
	} else {
		printf("==== CPU mode: time for initializing the data: %f.\n", (iStop - iStart)*totalLoop);
	}

	printf("===== add data begins...\n");
	iStart = currentTimeCPUSecond();
	{
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((nElement+threadsPerBlock.x-1) / threadsPerBlock.x);
		vecAdd<<<numBlocks, threadsPerBlock>>>(memoryOnDevice_A, memoryOnDevice_B, memoryOnDevice_C);
		//cudaMemcpy(g_C[loop], g_A[loop], nElem, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
	iStop = currentTimeCPUSecond();
	printf("==== Time for adding the data: %f.\n", (iStop - iStart)*totalLoop);

	cudaFree(memoryOnDevice_A);
	cudaFree(memoryOnDevice_B);
	cudaFree(memoryOnDevice_C);
}

void profileAddtion (unsigned int nElement, unsigned int totalLoop) {
	unsigned int nBytes = nElement * sizeof(float);

	if (nBytes < 1024) {
		printf("Allocate nbytes is %d B.\n", nBytes);
	} else if (nBytes >= 1024 && nBytes < 1024*1024) {
		printf("Allocate nbytes is %d KB.\n", nBytes/1024);
	} else {
		printf("Allocate nbytes is %d MB.\n", nBytes/(1024*1024));
	}

	// allocate memory
	float *g_A[64*1024], *g_B[64*1024], *g_C[64*1024];
	for (int loop=0; loop<totalLoop; loop++) {
		// unsigned int flags = cudaMemAttachHost;
		unsigned int flags = cudaMemAttachGlobal;
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_A[loop], nBytes, flags));
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_B[loop], nBytes, flags));
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_C[loop], nBytes, flags));
	}

	printf("===== inital data begins...\n");
	int mode = initializingMode_;
	double iStart = currentTimeCPUSecond();
	for (int loop=0; loop<totalLoop; loop++) {
		initialData(g_A[loop], nElement, 2.0f, mode);
		initialData(g_B[loop], nElement, 2.0f, mode);
		initialData(g_C[loop], nElement, 0.0f, mode);
	}
	double iStop = currentTimeCPUSecond();
	if (mode == 0) {
		printf("==== GPU mode: time for initializing the data: %f.\n", iStop - iStart);
	} else {
		printf("==== CPU mode: time for initializing the data: %f.\n", iStop - iStart);
	}

	printf("===== add data begins...\n");
	iStart = currentTimeCPUSecond();
	for (int loop=0; loop<totalLoop; loop++) {
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((nElement+threadsPerBlock.x-1) / threadsPerBlock.x);
		vecAdd<<<numBlocks, threadsPerBlock>>>(g_A[loop], g_B[loop], g_C[loop]);
		//cudaMemcpy(g_C[loop], g_A[loop], nElem, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
	iStop = currentTimeCPUSecond();
	printf("==== Time for adding the data: %f.\n", iStop - iStart);

	//Check the accuracy
	float ans = 4.0f;
	//printf("===== ans is %f\n", ans);
	int ii, jj;
	for (ii = 0; ii < totalLoop; ii++) {
		for (jj = 0; jj < nElement; jj++) {
			if ((g_C[ii])[jj] != ans)
			{
				printf("Error happens, should enable DEBUG mode to investigate.\n");
				break;
			}
		}
	}

	if(ii==totalLoop && jj==nElement) {
		printf("Testing is passed.\n");
	}

#ifdef DEBUG
	printf("===== Check the results...\n");

	for (int i = 0; i < totalLoop; i++) {
		printf("\n======================================================\n");
		for (int j = 0; j < 8; j++) {
			//if ((g_C[i])[j] != ans)
			{
				printf("%3.0f ", (g_A[i])[j]);
			}
		}
	}
	printf("\n");

	float ans = 4.0f;
	printf("===== ans is %f\n", ans);
	for (int i = 0; i < totalLoop; i++) {
		printf("\n======================================================\n");
		for (int j = 0; j < 8; j++) {
			//if ((g_C[i])[j] != ans)
			{
				printf("%3.0f ", (g_C[i])[j]);
			}
		}
	}
	printf("\n");
#endif

	for (int i = 0; i < totalLoop; i++) {
		cudaFree(g_A[i]);
		cudaFree(g_B[i]);
		cudaFree(g_C[i]);
	}
}



void test_cudaMallocManaged(int dev, int ipower) {
	int val;

	// Check if supports managed memory
	CHECK_CUDA_RESULT(cudaDeviceGetAttribute(&val, cudaDevAttrManagedMemory, dev));

	// Check concurrent managed access, for cuda 8.0
	cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentManagedAccess, dev);
	if (!val) {
		printf("*** Warn: Concurrent managed access is not supported!\n");
	}

	int totalLoop = totalLoop_;

	profileAddtionBasement(dataBlockSize_, totalLoop);
	profileAddtion(dataBlockSize_, totalLoop);
}

int main(int argc, char* argv[]) {

	// set up device
	int dev = 0;
	cudaSetDevice(dev);

	// get device properties
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	// check uva supporting
	if (deviceProp.unifiedAddressing) {
		printf("Device %d supports uva memory!\n", dev);
	} else {
		printf("Device %d does not support uva memory!\n", dev);
		exit(EXIT_SUCCESS);
	}

	// set up date size of vectors
	int ipower = 20+4;
	if (argc > 1)
		ipower = atoi(argv[1]);

	test_cudaMallocManaged(dev, ipower);


	cudaDeviceReset();
}

