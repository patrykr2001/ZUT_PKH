#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/time.h>
extern "C" {
#include "qdbmp.h"
}

__device__ void insertion_sort(UCHAR* values, int length) {
    for (int i = 1; i < length; ++i) {
        UCHAR key = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j];
            j = j - 1;
        }
        values[j + 1] = key;
    }
}

__global__ void apply_median_filter_gpu(UCHAR* d_input, UCHAR* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        UCHAR r_values[9], g_values[9], b_values[9];
        int idx = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int pixel_idx = 3 * ((y + dy) * width + (x + dx));
                r_values[idx] = d_input[pixel_idx];
                g_values[idx] = d_input[pixel_idx + 1];
                b_values[idx] = d_input[pixel_idx + 2];
                idx++;
            }
        }
        insertion_sort(r_values, 9);
        insertion_sort(g_values, 9);
        insertion_sort(b_values, 9);
        int output_idx = 3 * (y * width + x);
        d_output[output_idx] = r_values[4];
        d_output[output_idx + 1] = g_values[4];
        d_output[output_idx + 2] = b_values[4];
    }
}

void apply_median_filter_gpu_wrapper(BMP* bmp) {
    UINT width = BMP_GetWidth(bmp);
    UINT height = BMP_GetHeight(bmp);
    UCHAR* h_input = (UCHAR*)malloc(3 * width * height * sizeof(UCHAR));
    UCHAR* h_output = (UCHAR*)malloc(3 * width * height * sizeof(UCHAR));

    for (UINT x = 0; x < width; ++x) {
        for (UINT y = 0; y < height; ++y) {
            UCHAR r, g, b;
            BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);
            int idx = 3 * (y * width + x);
            h_input[idx] = r;
            h_input[idx + 1] = g;
            h_input[idx + 2] = b;
        }
    }

    UCHAR *d_input, *d_output;
    cudaMalloc(&d_input, 3 * width * height * sizeof(UCHAR));
    cudaMalloc(&d_output, 3 * width * height * sizeof(UCHAR));
    cudaMemcpy(d_input, h_input, 3 * width * height * sizeof(UCHAR), cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                         (height + threads_per_block.y - 1) / threads_per_block.y);

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    apply_median_filter_gpu<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    gettimeofday(&end_time, NULL);
    double elapsed_gpu = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) * 1e-6;
    std::cout << "GPU time: " << elapsed_gpu << " seconds" << std::endl;

    cudaMemcpy(h_output, d_output, 3 * width * height * sizeof(UCHAR), cudaMemcpyDeviceToHost);

    BMP* output = BMP_Create(width, height, BMP_GetDepth(bmp));
    for (UINT x = 0; x < width; ++x) {
        for (UINT y = 0; y < height; ++y) {
            int idx = 3 * (y * width + x);
            BMP_SetPixelRGB(output, x, y, h_output[idx], h_output[idx + 1], h_output[idx + 2]);
        }
    }

    BMP_WriteFile(output, "output_gpu.bmp");
    BMP_Free(output);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

bool compare_bmp_files(const char* file1, const char* file2) {
    BMP* bmp1 = BMP_ReadFile(file1);
    BMP* bmp2 = BMP_ReadFile(file2);

    if (BMP_GetError() != BMP_OK) {
        std::cerr << "Error reading BMP files!" << std::endl;
        return false;
    }

    UINT width1 = BMP_GetWidth(bmp1);
    UINT height1 = BMP_GetHeight(bmp1);
    UINT width2 = BMP_GetWidth(bmp2);
    UINT height2 = BMP_GetHeight(bmp2);

    if (width1 != width2 || height1 != height2) {
        BMP_Free(bmp1);
        BMP_Free(bmp2);
        return false;
    }

    for (UINT x = 0; x < width1; ++x) {
        for (UINT y = 0; y < height1; ++y) {
            UCHAR r1, g1, b1, r2, g2, b2;
            BMP_GetPixelRGB(bmp1, x, y, &r1, &g1, &b1);
            BMP_GetPixelRGB(bmp2, x, y, &r2, &g2, &b2);
            if (r1 != r2 || g1 != g2 || b1 != b2) {
                BMP_Free(bmp1);
                BMP_Free(bmp2);
                return false;
            }
        }
    }

    BMP_Free(bmp1);
    BMP_Free(bmp2);
    return true;
}

void apply_median_filter_cpu(BMP* bmp) {
    UINT width = BMP_GetWidth(bmp);
    UINT height = BMP_GetHeight(bmp);
    BMP* output = BMP_Create(width, height, BMP_GetDepth(bmp));

    for (UINT x = 1; x < width - 1; ++x) {
        for (UINT y = 1; y < height - 1; ++y) {
            std::vector<UCHAR> r_values, g_values, b_values;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    UCHAR r, g, b;
                    BMP_GetPixelRGB(bmp, x + dx, y + dy, &r, &g, &b);
                    r_values.push_back(r);
                    g_values.push_back(g);
                    b_values.push_back(b);
                }
            }
            std::sort(r_values.begin(), r_values.end());
            std::sort(g_values.begin(), g_values.end());
            std::sort(b_values.begin(), b_values.end());
            BMP_SetPixelRGB(output, x, y, r_values[4], g_values[4], b_values[4]);
        }
    }

    BMP_WriteFile(output, "output_cpu.bmp");
    BMP_Free(output);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input BMP file>" << std::endl;
        return 1;
    }

    BMP* bmp = BMP_ReadFile(argv[1]);
    if (BMP_GetError() != BMP_OK) {
        std::cerr << "Error reading BMP file!" << std::endl;
        return 1;
    }

    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    apply_median_filter_cpu(bmp);
    gettimeofday(&end_time, NULL);
    double elapsed_cpu = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) * 1e-6;
    std::cout << "CPU time: " << elapsed_cpu << " seconds" << std::endl;

    apply_median_filter_gpu_wrapper(bmp);
    BMP_Free(bmp);

    bool identical = compare_bmp_files("output_cpu.bmp", "output_gpu.bmp");
    if (identical) {
        std::cout << "The output files are identical." << std::endl;
    } else {
        std::cout << "The output files are not identical." << std::endl;
    }

    return 0;
}