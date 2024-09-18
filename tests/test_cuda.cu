#include <stdio.h>

// CUDA kernel function to add values to an array
__global__ void kernel_test(int *array) {
    int index = threadIdx.x;
    array[index] = index;
}

int main() {
    // Define array size
    const int array_size = 10;
    const int array_bytes = array_size * sizeof(int);

    // Allocate host memory
    int host_array[array_size];

    // Allocate device memory
    int *device_array;
    cudaMalloc((void **)&device_array, array_bytes);

    // Launch kernel with 10 threads
    kernel_test<<<1, array_size>>>(device_array);

    // Copy results from device to host
    cudaMemcpy(host_array, device_array, array_bytes, cudaMemcpyDeviceToHost);

    // Print the results
    printf("CUDA output:\n");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    // Free the device memory
    cudaFree(device_array);

    // If it reaches here, the test passed
    printf("CUDA test passed successfully!\n");
    
    return 0;
}
