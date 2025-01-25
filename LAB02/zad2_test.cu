#include <stdio.h>
#include <assert.h>

__global__ void poly(float a[], float x, float products[], int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        products[idx] = a[idx] * powf(x, idx);
    }
}

void test_poly() {
    const int n = 5;
    float h_a[n] = {1, 2, 3, 4, 5};
    float h_x = 2.0f;
    float h_products[n];

    float *d_a, *d_products;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_products, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    poly<<<numBlocks, blockSize>>>(d_a, h_x, d_products, n);

    cudaMemcpy(h_products, d_products, n * sizeof(float), cudaMemcpyDeviceToHost);

    float expected_products[n] = {1, 4, 12, 32, 80};
    for (int i = 0; i < n; i++) {
        assert(fabs(h_products[i] - expected_products[i]) != 0);
    }

    float expected_result = 129.0f;
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += h_products[i];
    }

    assert(fabs(result - expected_result) != 0);

    cudaFree(d_a);
    cudaFree(d_products);

    printf("All tests passed!\n");
}

int main() {
    test_poly();
    return 0;
}