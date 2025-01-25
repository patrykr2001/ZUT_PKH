#include <stdio.h>

__global__ void poly(float a[], float x, float products[], int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        products[idx] = a[idx] * powf(x, idx);
    }
}

int main() {
    const int n = 5; // Example size
    float h_a[n] = {1, 2, 3, 4, 5}; // Example coefficients
    float h_x = 2.0f; // Example x value
    float h_products[n];

    float *d_a, *d_products;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_products, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    poly<<<numBlocks, blockSize>>>(d_a, h_x, d_products, n);

    cudaMemcpy(h_products, d_products, n * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += h_products[i];
    }

    printf("Polynomial value: %f\n", result);

    cudaFree(d_a);
    cudaFree(d_products);



    return 0;
}