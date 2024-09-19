#include <stdio.h>
// This is without UVM

/*
Debugging and Profiling
Tools:
cuda-gdb: A CUDA-aware extension of the GNU Debugger for debugging device code.
nvprof / Nsight Systems: Profiling tools to analyze performance bottlenecks.
Use device-side printf (available in newer CUDA versions) for debugging.
*/

// CUDA kernel function to add values to an array
#include <stdio.h>

// Device code (kernel) running on the GPU
__global__ void kernel_test(int *array) {
    int index = threadIdx.x; // Each thread gets its unique index
    array[index] = index;    // Writes its index value into the device array
}

int main() {
    // Host code running on the CPU
    const int array_size = 10;
    const int array_bytes = array_size * sizeof(int);

    // Host memory allocation
    int host_array[array_size];

    // Device memory allocation
    int *device_array;
    cudaMalloc((void **)&device_array, array_bytes);

    // Kernel launch: host instructs device to execute kernel
    kernel_test<<<1, array_size>>>(device_array);

    // Data transfer from device to host
    cudaMemcpy(host_array, device_array, array_bytes, cudaMemcpyDeviceToHost);

    // Output results on the host
    printf("CUDA output:\n");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    // Device memory deallocation
    cudaFree(device_array);

    // Host indicates successful completion
    printf("CUDA test passed successfully!\n");

    return 0;
}
