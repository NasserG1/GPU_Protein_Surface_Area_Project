#include <stdio.h>

__global__ void helloWorld() {
    printf("Hello, World from GPU!\n");
}

int main() {
    helloWorld<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
