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

__global__
void matrix_matrix_mult_kernel(float *A, float *B, float *C,
                               unsigned long int A_height,
                               unsigned long int A_width,
                               unsigned long int B_width,
                               unsigned long int total_elements)
{
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int stride = gridDim.x * blockDim.x;

    for (unsigned long int index = idx; index < total_elements; index += stride) {
        unsigned long int row = index / B_width;
        unsigned long int col = index % B_width;

        float sum = 0.0f;
        for (unsigned long int k = 0; k < A_width; ++k) {
            sum += A[row * A_width + k] * B[k * B_width + col];
        }
        C[index] = sum;
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

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixA->h_rows == NULL ||
        matrixB == NULL || matrixB->h_rows == NULL ||
        matrixC == NULL || matrixC->h_rows == NULL) {
        return 0;
    }

    if (matrixA->width != matrixB->height) {
        return 0;
    }

    unsigned long int total_elements = matrixC->height * matrixC->width;
    unsigned long int A_size = matrixA->height * matrixA->width * sizeof(float);
    unsigned long int B_size = matrixB->height * matrixB->width * sizeof(float);
    unsigned long int C_size = matrixC->height * matrixC->width * sizeof(float);

    cudaError_t err;

    // Aloca memória no device
    if ((err = cudaMalloc((void **)&matrixA->d_rows, A_size)) != cudaSuccess ||
        (err = cudaMalloc((void **)&matrixB->d_rows, B_size)) != cudaSuccess ||
        (err = cudaMalloc((void **)&matrixC->d_rows, C_size)) != cudaSuccess) {
        fprintf(stderr, "Erro ao alocar memória no device: %s\n", cudaGetErrorString(err));
        if (matrixA->d_rows) cudaFree(matrixA->d_rows);
        if (matrixB->d_rows) cudaFree(matrixB->d_rows);
        if (matrixC->d_rows) cudaFree(matrixC->d_rows);
        return 0;
    }

    // Copia dados para o device
    if ((err = cudaMemcpy(matrixA->d_rows, matrixA->h_rows, A_size, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (err = cudaMemcpy(matrixB->d_rows, matrixB->h_rows, B_size, cudaMemcpyHostToDevice)) != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar dados para o device: %s\n", cudaGetErrorString(err));
        cudaFree(matrixA->d_rows);
		cudaFree(matrixB->d_rows);
		cudaFree(matrixC->d_rows);
        return 0;
    }

    // Configuração de grid e blocos
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Lança o kernel
    matrix_matrix_mult_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->height, matrixA->width, matrixB->width, total_elements);

    // Sincroniza e verifica erros
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro durante execução do kernel: %s\n", cudaGetErrorString(err));
        cudaFree(matrixA->d_rows);
		cudaFree(matrixB->d_rows);
		cudaFree(matrixC->d_rows);
        return 0;
    }

    // Copia o resultado de volta
    err = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, C_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar resultado para o host: %s\n", cudaGetErrorString(err));
        cudaFree(matrixA->d_rows);
		cudaFree(matrixB->d_rows);
		cudaFree(matrixC->d_rows);
        return 0;
    }

    // Libera memória no device
    cudaFree(matrixA->d_rows);
    cudaFree(matrixB->d_rows);
    cudaFree(matrixC->d_rows);

    return 1;
}