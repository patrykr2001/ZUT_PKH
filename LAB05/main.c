#define CL_TARGET_OPENCL_VERSION 300
#define MATRIX_SIZE 500
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char* readKernelSource(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);
    char* source = (char*)malloc(*size + 1);
    fread(source, 1, *size, file);
    source[*size] = '\0';
    fclose(file);
    return source;
}

void matrixMultiplyCPU(float A[MATRIX_SIZE][MATRIX_SIZE], float B[MATRIX_SIZE][MATRIX_SIZE], float C[MATRIX_SIZE][MATRIX_SIZE], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int punkt1(){
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_int err;

    // Get platform count
    err = clGetPlatformIDs(0, NULL, &platformCount);
    if (err != CL_SUCCESS) {
        printf("Unable to get platform count\n");
        return 1;
    }

    // Get all platforms
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    err = clGetPlatformIDs(platformCount, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get platform IDs\n");
        free(platforms);
        return 1;
    }

    // For each platform, get all devices
    for (cl_uint i = 0; i < platformCount; i++) {
        // Get device count
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        if (err != CL_SUCCESS) {
            printf("Unable to get device count for platform %u\n", i);
            continue;
        }

        // Get all devices
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
        if (err != CL_SUCCESS) {
            printf("Unable to get device IDs for platform %u\n", i);
            free(devices);
            continue;
        }

        // Print device names
        for (cl_uint j = 0; j < deviceCount; j++) {
            char deviceName[128];
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            if (err == CL_SUCCESS) {
                printf("Platform %u, Device %u: %s\n", i, j, deviceName);
            } else {
                printf("Unable to get device name for platform %u, device %u\n", i, j);
            }
        }

        free(devices);
    }

    free(platforms);
    return 0;
}

int punkt2(){
    int n = MATRIX_SIZE;
    size_t bytes = n * n * sizeof(float);

    float A[n][n];
    float B[n][n];
    float C[n][n];

    // Initialize matrices A and B with values from 1 to n*n
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i * n + j + 1;
            B[i][j] = i * n + j + 1;
        }
    }

    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    cl_int err;

    // Get platform count
    err = clGetPlatformIDs(0, NULL, &platformCount);
    if (err != CL_SUCCESS) {
        printf("Unable to get platform count\n");
        return 1;
    }
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    err = clGetPlatformIDs(platformCount, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get platform IDs\n");
        free(platforms);
        return 1;
    }

    // Get device
    err = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get device ID for platform %p\n", platforms[2]);
        return 1;
    }

    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create context\n");
        return 1;
    }
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create command queue\n");
        clReleaseContext(context);
        return 1;
    }

    // Create buffers
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create buffer for A\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create buffer for B\n");
        clReleaseMemObject(d_A);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create buffer for C\n");
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Write data to buffers
    err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to write to buffer A\n");
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    err = clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, bytes, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to write to buffer B\n");
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Read kernel source from file
    size_t kernelSize;
    char* kernelSource = readKernelSource("matrix_multiplication.cl", &kernelSize);

    // Create program and kernel
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernelSize, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create program\n");
        free(kernelSource);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to build program\n");
        free(kernelSource);
        clReleaseProgram(program);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    kernel = clCreateKernel(program, "matrix_multiply", &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create kernel\n");
        free(kernelSource);
        clReleaseProgram(program);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Execute kernel
    size_t globalSize[2] = {n, n};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to enqueue kernel\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(kernelSource);
        return 1;
    }

    // Read result
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to read buffer C\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(kernelSource);
        return 1;
    }

    // Print result
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", C[i][j]);
    //     }
    //     printf("\n");
    // }

    // Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernelSource);

    return 0;
}

int punkt3(){
    // CPU matrix multiplication
    int n = MATRIX_SIZE;
    float A[n][n];
    float B[n][n];
    float C[n][n];

    // Initialize matrices A and B with values from 1 to n*n
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i * n + j + 1;
            B[i][j] = i * n + j + 1;
        }
    }

    matrixMultiplyCPU(A, B, C, n);

    return 0;
}

int main() {
    int v = punkt1();
    if(v != 0){
        printf("Error\n");
        return v;
    }

    struct timespec startGpu, endGpu;
    clock_gettime(CLOCK_MONOTONIC, &startGpu);

    v = punkt2();
    if(v != 0){
        printf("Error\n");
        return v;
    }

    clock_gettime(CLOCK_MONOTONIC, &endGpu);
    double gpu_time = (endGpu.tv_sec - startGpu.tv_sec) + (endGpu.tv_nsec - startGpu.tv_nsec) / 1e9;
    printf("GPU time: %f seconds\n", gpu_time);

    struct timespec startCpu, endCpu;
    clock_gettime(CLOCK_MONOTONIC, &startCpu);

    v = punkt3();
    if(v != 0){
        printf("Error\n");
        return v;
    }

    clock_gettime(CLOCK_MONOTONIC, &endCpu);
    double cpu_time = (endCpu.tv_sec - startCpu.tv_sec) + (endCpu.tv_nsec - startCpu.tv_nsec) / 1e9;
    printf("CPU time: %f seconds\n", cpu_time);

    double time_difference = cpu_time - gpu_time;
    printf("Time difference (CPU - GPU): %f seconds\n", time_difference);

    double percentage_faster = ((cpu_time - gpu_time) / cpu_time) * 100;
    printf("GPU is %.2f%% faster than CPU\n", percentage_faster);

    return 0;
}