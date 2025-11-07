#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"
extern "C" {
#include "timer.h"
}

__global__ 
void scalar_matrix_mult_kernel(float scalar_value, float *matrix_data, unsigned long int total_elements) {
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long int stride = gridDim.x * blockDim.x;
    for (unsigned long int i = idx; i < total_elements; i += stride) {
        matrix_data[i] *= scalar_value;
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->h_rows == NULL) {
        return 0;
    }

    unsigned long int total_elements = matrix->height * matrix->width;
	cudaError_t cudaError;

    // Aloca memória no device
    cudaError = cudaMalloc(&matrix->d_rows, total_elements * sizeof(float));
    if (cudaError != cudaSuccess) {
        printf("Erro ao alocar memória no device: %s\n", cudaGetErrorString(cudaError));
		return 0;
    }

    // Copia os dados da matriz do host para o device
    cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        printf("Erro ao copiar dados para o device: %s\n", cudaGetErrorString(cudaError));
        cudaFree(matrix->d_rows);
        return 0;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Lança o kernel
    scalar_matrix_mult_kernel<<<blocks_per_grid, threads_per_block>>>(
		scalar_value, matrix->d_rows, total_elements
	);

    // Sincroniza e checa erro de execução
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("Erro durante execução do kernel: %s\n", cudaGetErrorString(cudaError));
        cudaFree(matrix->d_rows);
        return 0;
    }

    // Copia o resultado de volta para a matriz original
    cudaError = cudaMemcpy(matrix->h_rows, matrix->d_rows, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        printf("Erro ao copiar resultado para o host: %s\n", cudaGetErrorString(cudaError));
        cudaFree(matrix->d_rows);
        return 0;
    }

    cudaFree(matrix->d_rows);

    return 1;
}