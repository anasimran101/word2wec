#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "skipgram.h"

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}


void trainGpu(int sentence_index) {};
void getResultData() {};
void initGpu() {
	// Device query
	int nDevices;
	cudaCheck(cudaGetDeviceCount(&nDevices));
    printf("Number of CUDA devices: %d\n", nDevices);
	for (int i = 0; i < nDevices; i++) {
	    cudaDeviceProp prop;
	    cudaCheck(cudaGetDeviceProperties(&prop, i));
	    printf("Device Number: %d\n", i);
	    printf("  Device name: %s\n", prop.name);
	    printf("  Memory Clock Rate (KHz): %d\n",
	           prop.memoryClockRate);
	    printf("  Memory Bus Width (bits): %d\n",
	           prop.memoryBusWidth);
	    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
	           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}
	int device = 0;
	cudaCheck(cudaSetDevice(device));
	cudaDeviceProp prop;
	cudaCheck(cudaGetDeviceProperties(&prop, device));


}
void freeGpu(){};