#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

int N = 1000;

struct WIERZCHOLEK {
    float x, y;
};

__global__ void obracanie(WIERZCHOLEK *W, float alfa, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        float x = W[i].x * cos(alfa) - W[i].y * sin(alfa);
        float y = W[i].x * sin(alfa) + W[i].y * cos(alfa);
        W[i].x = x;
        W[i].y = y;
    }
}

void generateRandomFloats(float *num1, float *num2) {
    *num1 = (float)rand() / RAND_MAX;
    *num2 = (float)rand() / RAND_MAX;
}

void fillWithRandomData(WIERZCHOLEK *W1, WIERZCHOLEK *W2) {
    for (int i = 0; i < N; i++) {
        float x, y;
        generateRandomFloats(&x, &y);
        W1[i].x = W2[i].x = x;
        W1[i].y = W2[i].y = y;
    }
}

int main(void) {
    srand(time(NULL));

    struct timeval start_time_cpu, end_time_cpu, start_time_gpu, end_time_gpu, start_time_gpuio, end_time_gpuio;
    long seconds, microseconds;
    double elapsed;
    double cpuTimes[25];
    double gpuTimes[25];
    double gpuIOTimes[25];

    float alfa = 3.1415;
    int ile_cudow;
    int k = 0;

    struct WIERZCHOLEK *Fig = (WIERZCHOLEK *)calloc(10, sizeof(WIERZCHOLEK));
    WIERZCHOLEK *Wyn;
    cudaHostRegister(Fig, 10 * sizeof(WIERZCHOLEK), cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&Wyn, Fig, 0);

    int watki_na_blok = 1024;
    int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;

    obracanie<<<bloki_na_siatke, watki_na_blok>>>(Wyn, alfa, 10);
    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
    cudaDeviceSynchronize();

    WIERZCHOLEK *d_W;

    for (int i = 1000; i <= 10000000; i *= 10) {
        N = i;
        for (int j = 1; j <= 5; j++) {
            struct WIERZCHOLEK *Figura1 = (WIERZCHOLEK *)calloc(N, sizeof(WIERZCHOLEK));
            struct WIERZCHOLEK *Figura2 = (WIERZCHOLEK *)calloc(N, sizeof(WIERZCHOLEK));

            fillWithRandomData(Figura1, Figura2);

            printf("Rozmiar wektora: %i\nWykonanie: %i\n", N, j);
            puts("--- CPU --- ");

            gettimeofday(&start_time_cpu, NULL);

            for (int i = 0; i < N; i++) {
                float x = Figura1[i].x * cos(alfa) - Figura1[i].y * sin(alfa);
                float y = Figura1[i].x * sin(alfa) + Figura1[i].y * cos(alfa);
                Figura1[i].x = x;
                Figura1[i].y = y;
            }

            gettimeofday(&end_time_cpu, NULL);

            seconds = end_time_cpu.tv_sec - start_time_cpu.tv_sec;
            microseconds = end_time_cpu.tv_usec - start_time_cpu.tv_usec;
            elapsed = seconds + microseconds * 1e-6;
            printf("Czas CPU: %.6f sekund\n", elapsed);
            cpuTimes[k] = elapsed;

            cudaGetDeviceCount(&ile_cudow);
            if (ile_cudow == 0) {
                perror("Nie ściemniaj – nie masz CUDY");
                return 1;
            }
            puts("--- GPU ---");

            gettimeofday(&start_time_gpuio, NULL);

            cudaHostRegister(Figura2, N * sizeof(WIERZCHOLEK), cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&d_W, Figura2, 0);

            int watki_na_blok = 1024;
            int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;

            gettimeofday(&start_time_gpu, NULL);

            obracanie<<<bloki_na_siatke, watki_na_blok>>>(d_W, alfa, N);
            cudaError_t err = cudaGetLastError();
            assert(err == cudaSuccess);
            cudaDeviceSynchronize();

            gettimeofday(&end_time_gpu, NULL);

            cudaHostUnregister(Figura2);

            gettimeofday(&end_time_gpuio, NULL);

            seconds = end_time_gpu.tv_sec - start_time_gpu.tv_sec;
            microseconds = end_time_gpu.tv_usec - start_time_gpu.tv_usec;
            elapsed = seconds + microseconds * 1e-6;
            printf("Czas GPU: %.6f sekund\n", elapsed);
            gpuTimes[k] = elapsed;

            seconds = end_time_gpuio.tv_sec - start_time_gpuio.tv_sec;
            microseconds = end_time_gpuio.tv_usec - start_time_gpuio.tv_usec;
            elapsed = seconds + microseconds * 1e-6;
            gpuIOTimes[k++] = elapsed;
            printf("Czas GPU z IO: %.6f sekund\n", elapsed);

            free(Figura1);
            free(Figura2);
        }
    }

    double cpuMean[5];
    double gpuMean[5];
    double gpuIOMean[5];
    int transferCost[5];
    k = 0;

    for (int i = 0; i < 5; i++) {
        cpuMean[i] = 0;
        gpuMean[i] = 0;
        gpuIOMean[i] = 0;

        for (int j = 0; j < 5; j++) {
            cpuMean[i] += cpuTimes[k];
            gpuMean[i] += gpuTimes[k];
            gpuIOMean[i] += gpuIOTimes[k++];
        }

        cpuMean[i] /= 5;
        gpuMean[i] /= 5;
        gpuIOMean[i] /= 5;
        double cost = gpuIOMean[i] / gpuMean[i] * 100;
        transferCost[i] = (int)cost;
    }

    printf("\n\n\n\n\n");

    printf("Średnie czasu dla poszczególnych wielkości wektora:\n");
    printf("\t\tCPU (s)\t\tGPU (s)\t\tGPU z IO (s)\tKoszt transferu IO (%)\n");
    int ind = 1000;
    for (int i = 0; i < 5; i++) {
        printf("%10d:\t%f\t%f\t%f\t%5d%%\n", ind, cpuMean[i], gpuMean[i], gpuIOMean[i], transferCost[i]);
        ind *= 10;
    }

    return 0;
}