#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    printf("Detected %d CUDA Capable device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Total amount of global memory: %.2f MB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024));
        printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Clock rate: %.0f MHz\n", deviceProp.clockRate * 1e-3f);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Can map host memory: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
    }

    return 0;
}