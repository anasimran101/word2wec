#include <iostream>

__global__ void hello_cuda() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

int main() {
    hello_cuda<<<1, 10>>>();
    cudaDeviceSynchronize();
    std::cout << "CUDA kernel executed.\n";
    return 0;
}