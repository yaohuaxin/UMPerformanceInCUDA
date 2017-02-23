/*****************************************************
 * This file tests cuda memory management APIs.
 *****************************************************/
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(float* A, float* B, float* C) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
	//printf("From GPU %f.\n", C[i]);
}

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

void initialData(float *h, long long n, float data) {
	long long i;
	for (i = 0; i < n; i++) {
		h[i] = data;
	}
}

void test_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
	cudaDeviceGetAttribute(value, attr, device);
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

	// Calculate number of elements and bytes
	long long nElem = ((long long) 1) << ipower;
	long long nBytes = nElem * sizeof(float);

	// allocate memory
	float *g_A[64*1024], *g_B[64*1024], *g_C[64*1024];

	int totalLoop = 1024*2;
	//int totalLoop = 10;

	for (int loop=0; loop<totalLoop; loop++) {
		printf("==== ==== ==== ==== Loop: %d.\n", loop);
		if (ipower < 18) {
			printf("Vector size is %lld, nbytes is %f KB\n", nElem,
					(float) nBytes / (1024.0f));
		} else {
			printf("Vector size is %lld, nbytes is %f MB\n", nElem,
					(float) nBytes / (1024.0f * 1024.0f));
		}
		// unsigned int flags = cudaMemAttachHost;
		unsigned int flags = cudaMemAttachGlobal;
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_A[loop], nBytes, flags));
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_B[loop], nBytes, flags));
		CHECK_CUDA_RESULT(cudaMallocManaged(&g_C[loop], nBytes, flags));

		printf("===== inital data begins...\n");
		initialData(g_A[loop], nElem, 2.0f);
		initialData(g_B[loop], nElem, 2.0f);

		printf("===== synchronize begins...\n");
		cudaDeviceSynchronize();

		printf("===== add data begins...\n");
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((nElem+threadsPerBlock.x-1) / threadsPerBlock.x);
		printf("===== numBlocks is %d, threadsPerBlock is %d\n", numBlocks.x, threadsPerBlock.x);
		// Kernel invocation with N threads
		vecAdd<<<numBlocks, threadsPerBlock>>>(g_A[loop], g_B[loop], g_C[loop]);
		//cudaMemcpy(g_C[loop], g_A[loop], nElem, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

	}

	printf("===== Check the results...\n");

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
	cudaFree(g_A);
	cudaFree(g_B);
	cudaFree(g_C);

	cudaDeviceReset();
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

}

