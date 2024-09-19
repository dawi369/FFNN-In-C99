#include <stdio.h>
#include <cuda_runtime.h>
// This is with UVM

__global__ void kernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx * idx;
}

int main() {
    const int N = 10;
    int *data;

    // Allocate Unified Memory
    cudaError_t err = cudaMallocManaged(&data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating Unified Memory: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel on the GPU
    kernel<<<1, N>>>(data);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Access data on the CPU
    printf("Unified Memory Test Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    // Free Unified Memory
    cudaFree(data);

    return 0;
}
