#include <iostream>

__global__ void hello_cuda() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

int main() {
    hello_cuda<<<1, 10>>>(); // launch 1 block with 10 threads
    cudaDeviceSynchronize(); // wait for GPU to finish
    std::cout << "CUDA kernel executed.\n";
    return 0;
}
