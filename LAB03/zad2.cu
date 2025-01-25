#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/time.h>
#include <cuda.h>

double calculate_pi_sequential(double eps, int &iterations) {
    double pi = 1.0;
    double lastPi = 2.0;
    double term;
    iterations = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    while (std::fabs(pi - lastPi) >= eps) {
        lastPi = pi;
        iterations++;
        term = (2.0 * iterations) / (2.0 * iterations - 1.0) * (2.0 * iterations) / (2.0 * iterations + 1.0);
        pi *= term;
    }

    gettimeofday(&end_time, NULL);
    double elapsed_sequential = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) * 1e-6;

    std::cout << "Sequential π: " << std::setprecision(20) << pi * 2 << std::endl;
    std::cout << "Sequential iterations: " << iterations << std::endl;
    std::cout << "Sequential time: " << elapsed_sequential << " seconds" << std::endl;

    return pi * 2;
}

__global__ void calculate_terms(double *terms, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iterations) {
        terms[idx] = (2.0 * (idx + 1)) / (2.0 * (idx + 1) - 1.0) * (2.0 * (idx + 1)) / (2.0 * (idx + 1) + 1.0);
    }
}

double calculate_pi_parallel(int iterations) {
    // double *terms;
    double *d_terms;
    double *h_terms = (double *)malloc(iterations * sizeof(double));

    struct timeval start_time, end_time;
    

    // cudaMallocManaged(&terms, iterations * sizeof(double));
    cudaMalloc(&d_terms, iterations * sizeof(double));
    // cudaMemcpy(d_terms, h_terms, iterations * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int blocks_per_grid = (iterations + threads_per_block - 1) / threads_per_block;

    // calculate_terms<<<blocks_per_grid, threads_per_block>>>(terms, iterations);
    gettimeofday(&start_time, NULL);
    calculate_terms<<<blocks_per_grid, threads_per_block>>>(d_terms, iterations);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_terms, d_terms, iterations * sizeof(double), cudaMemcpyDeviceToHost);
    
    double pi = 1.0;
    for (int i = 0; i < iterations; i++) {
        // pi *= terms[i];
        pi *= h_terms[i];
    }

    gettimeofday(&end_time, NULL);
    double elapsed_parallel = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) * 1e-6;

    
    std::cout << std::setprecision(20) << "Parallel π: " << pi * 2 << std::endl;
    std::cout << "Parallel time: " << elapsed_parallel << " seconds" << std::endl;

    // cudaFree(terms);
    cudaFree(d_terms);
    return pi * 2;
}


int main() {
    double eps;
    std::cout << "Enter the value of eps: ";
    std::cin >> eps;

    int iterations;
    double pi_sequential = calculate_pi_sequential(eps, iterations);
    double pi_parallel = calculate_pi_parallel(iterations);

    return 0;
}