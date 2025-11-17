#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"
extern "C" {
#include "timer.h"
}

static int g_threads_per_block   = 256;
static int g_max_blocks_per_grid = 4096;

__global__ 
void scalar_matrix_mult_kernel(float scalar_value, float *matrix_data, unsigned long int total_elements) {
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        matrix_data[idx] *= scalar_value;
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

__global__
void matrix_matrix_mult_partial_kernel(
    float *Arow,
    float *matrixB,
    float *Crow,
    unsigned long int A_width,
    unsigned long int B_width) {

    unsigned long int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= B_width)
        return;

    float sum = 0.0f;
    for (unsigned long int k = 0; k < A_width; ++k) {
        sum += Arow[k] * matrixB[k * B_width + col];
    }
    Crow[col] = sum;
}

int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    int dev = 0;
    cudaError_t err;

    // Tenta descobrir qual device está em uso
    err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        // Se algo der errado, volta pros defaults
        g_threads_per_block   = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    int maxThreadsPerBlock = 0;
    int maxGridDimX        = 0;

    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, dev);
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, dev);

    // Valida parâmetros
    if (threads_per_block <= 0 || threads_per_block > maxThreadsPerBlock ||
        max_blocks_per_grid <= 0 || max_blocks_per_grid > maxGridDimX) {

        g_threads_per_block   = 256;
        g_max_blocks_per_grid = 4096;
        return 0;
    }

    g_threads_per_block   = threads_per_block;
    g_max_blocks_per_grid = max_blocks_per_grid;
    return 1;
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->h_rows == NULL) {
        return 0;
    }

    unsigned long int total_elements = matrix->height * matrix->width;
	cudaError_t cudaError;

    if (matrix->alloc_mode == FULL_ALLOCATION) {
        cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows,
                            total_elements * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess) {
            printf("Erro ao copiar dados para o device (scalar FULL): %s\n", cudaGetErrorString(cudaError));
            return 0;
        }
        unsigned long int blocks_per_grid =
            (total_elements + g_threads_per_block - 1) / g_threads_per_block;

        if (blocks_per_grid == 0) {
            blocks_per_grid = 1;
        }
        if (blocks_per_grid > (unsigned long int)g_max_blocks_per_grid) {
            blocks_per_grid = g_max_blocks_per_grid;
        }

        // Lança o kernel
        scalar_matrix_mult_kernel<<<blocks_per_grid, g_threads_per_block>>>(
            scalar_value, matrix->d_rows, total_elements
        );

        // Sincroniza e checa erro de execução
        cudaError = cudaDeviceSynchronize();
        if (cudaError != cudaSuccess) {
            printf("Erro durante execução do kernel: %s\n", cudaGetErrorString(cudaError));
            return 0;
        }

        // Copia o resultado de volta para a matriz original
        cudaError = cudaMemcpy(matrix->h_rows, matrix->d_rows, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaError != cudaSuccess) {
            printf("Erro ao copiar resultado para o host: %s\n", cudaGetErrorString(cudaError));
            cudaFree(matrix->d_rows);
            return 0;
        }

        return 1;
    }
    else if (matrix->alloc_mode == PARTIAL_ALLOC) {
        unsigned long int row_elems = matrix->width;
        size_t row_bytes = row_elems * sizeof(float);

        for (unsigned long int row = 0; row < matrix->height; row++) {
            cudaError = cudaMemcpy(matrix->d_rows,
                                &matrix->h_rows[row * row_elems],
                                row_bytes,
                                cudaMemcpyHostToDevice);
            if (cudaError != cudaSuccess) {
                printf("Erro ao copiar linha A para o device (scalar PARTIAL): %s\n",
                    cudaGetErrorString(cudaError));
                return 0;
            }

            // executar kernel para 1 linha
            unsigned long int blocks_per_grid =
                (row_elems + g_threads_per_block - 1) / g_threads_per_block;

            if (blocks_per_grid > (unsigned long)g_max_blocks_per_grid)
                blocks_per_grid = g_max_blocks_per_grid;

            scalar_matrix_mult_kernel<<<blocks_per_grid, g_threads_per_block>>>(
                scalar_value, matrix->d_rows, row_elems
            );

            cudaError = cudaDeviceSynchronize();
            if (cudaError != cudaSuccess) {
                printf("Erro durante execução do kernel (scalar PARTIAL): %s\n",
                    cudaGetErrorString(cudaError));
                return 0;
            }

            // copiar linha de volta
            cudaError = cudaMemcpy(&matrix->h_rows[row * row_elems],
                                matrix->d_rows, row_bytes,
                                cudaMemcpyDeviceToHost);
            if (cudaError != cudaSuccess) {
                printf("Erro ao copiar linha A do device para host (scalar PARTIAL): %s\n",
                    cudaGetErrorString(cudaError));
                return 0;
            }
        }
        return 1;
    }
    return 0;
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

    unsigned long int A_size = matrixA->height * matrixA->width * sizeof(float);
    unsigned long int B_size = matrixB->height * matrixB->width * sizeof(float);
    unsigned long int C_size = matrixC->height * matrixC->width * sizeof(float);

    cudaError_t err;

    if (matrixA->alloc_mode == FULL_ALLOCATION &&
        matrixB->alloc_mode == FULL_ALLOCATION &&
        matrixC->alloc_mode == FULL_ALLOCATION) {
        
        // copia os dados da matriz do host para o device
        err = cudaMemcpy(matrixA->d_rows, matrixA->h_rows, A_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("cudaMemcpy HtoD matrixA FULL error: %s\n", cudaGetErrorString(err));
            return 0;
        }
        err = cudaMemcpy(matrixB->d_rows, matrixB->h_rows, B_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("cudaMemcpy HtoD matrixB FULL error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        unsigned long int total_elements = matrixC->height * matrixC->width;
        unsigned long int blocks_per_grid =
            (total_elements + g_threads_per_block - 1) / g_threads_per_block;

        if (blocks_per_grid == 0) blocks_per_grid = 1;
        if (blocks_per_grid > (unsigned long)g_max_blocks_per_grid)
            blocks_per_grid = g_max_blocks_per_grid;

        matrix_matrix_mult_kernel<<<blocks_per_grid, g_threads_per_block>>>(
            matrixA->d_rows, matrixB->d_rows, matrixC->d_rows,
            matrixA->height, matrixA->width, matrixB->width,
            total_elements
        );

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel matrix FULL error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        // copia os dados da matriz do device para o host
        err = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, C_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("cudaMemcpy DtoH matrixC FULL error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        return 1;
    }

    if (matrixA->alloc_mode == PARTIAL_ALLOC &&
        matrixB->alloc_mode == FULL_ALLOCATION &&
        matrixC->alloc_mode == PARTIAL_ALLOC)
    {
        // Copia matrixB inteira uma única vez
        err = cudaMemcpy(matrixB->d_rows, matrixB->h_rows, B_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("cudaMemcpy HtoD matrixB PARTIAL error: %s\n", cudaGetErrorString(err));
            return 0;
        }

        unsigned long int row_elems_A = matrixA->width;
        unsigned long int row_elems_C = matrixC->width;

        size_t row_bytes_A = row_elems_A * sizeof(float);
        size_t row_bytes_C = row_elems_C * sizeof(float);

        for (unsigned long int row = 0; row < matrixA->height; ++row) {
            // host → device: linha de matrixA
            err = cudaMemcpy(matrixA->d_rows,
                             &matrixA->h_rows[row * row_elems_A],
                             row_bytes_A,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("cudaMemcpy HtoD Arow PARTIAL error: %s\n", cudaGetErrorString(err));
                return 0;
            }

            // configura grid para 1 linha de matrixC
            unsigned long int blocks_per_grid =
                (row_elems_C + g_threads_per_block - 1) / g_threads_per_block;
            if (blocks_per_grid == 0) blocks_per_grid = 1;
            if (blocks_per_grid > (unsigned long)g_max_blocks_per_grid)
                blocks_per_grid = g_max_blocks_per_grid;

            matrix_matrix_mult_partial_kernel<<<blocks_per_grid, g_threads_per_block>>>(
                matrixA->d_rows,
                matrixB->d_rows,
                matrixC->d_rows,
                matrixA->width,
                matrixB->width
            );

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("kernel matrix PARTIAL error: %s\n", cudaGetErrorString(err));
                return 0;
            }

            // device → host: linha de matrixC
            err = cudaMemcpy(&matrixC->h_rows[row * row_elems_C],
                             matrixC->d_rows,
                             row_bytes_C,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                printf("cudaMemcpy DtoH Crow PARTIAL error: %s\n", cudaGetErrorString(err));
                return 0;
            }
        }
        return 1;
    }
    return 0;
}