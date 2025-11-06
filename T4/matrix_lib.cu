#include "matrix_lib.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ 
void scalar_matrix_mult_kernel(float scalar_value, float *matrix_data, unsigned long int total_elements) {
    unsigned long int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long int stride = gridDim.x * blockDim.x;
    for (unsigned long int i = idx; i < total_elements; i += stride) {
        matrix_data[i] *= scalar_value;
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }

    unsigned long int total_elements = (unsigned long int)matrix->height * matrix->width;
    float *d_matrix = NULL;

    // Aloca memória no device
    cudaError_t cudaError = cudaMalloc(&d_matrix, total_elements * sizeof(float));
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Erro ao alocar memória no device: %s\n", cudaGetErrorString(cudaError));
        return 0;
    }

    // Copia os dados da matriz do host para o device
    cudaError = cudaMemcpy(d_matrix, matrix->rows, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar dados para o device: %s\n", cudaGetErrorString(cudaError));
        cudaFree(d_matrix);
        return 0;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Lança o kernel
    scalar_matrix_mult_kernel<<<blocks_per_grid, threads_per_block>>>(scalar_value, d_matrix, total_elements);

    // Sincroniza e checa erro de execução
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Erro durante execução do kernel: %s\n", cudaGetErrorString(cudaError));
        cudaFree(d_matrix);
        return 0;
    }

    // Copia o resultado de volta para a matriz original
    cudaError = cudaMemcpy(matrix->rows, d_matrix, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar resultado para o host: %s\n", cudaGetErrorString(cudaError));
        cudaFree(d_matrix);
        return 0;
    }

    cudaFree(d_matrix);

    return 1;
}


int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC) {
    if (matrixA == NULL || matrixA->rows == NULL ||
        matrixB == NULL || matrixB->rows == NULL ||
        matrixC == NULL || matrixC->rows == NULL) {
        return 0;
    }

    if (matrixA->width != matrixB->height || 
        matrixC->height != matrixA->height || 
        matrixC->width != matrixB->width) {
        return 0;
    }
    
    //antes estava no arquivo matrix_lib_test.c, mas perecebi que faz mais sentido colocar aqui, ja q o processo de mult depende disso
    memset(matrixC->rows, 0, matrixC->height * matrixC->width * sizeof(float));

    pthread_t threads[NUM_THREADS];
    thread_data thread_args[NUM_THREADS];
    
    unsigned long int linhas_por_thread = matrixA->height / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {

        thread_args[i].matrixA = matrixA;
        thread_args[i].matrixB = matrixB;
        thread_args[i].matrixC = matrixC;
        thread_args[i].start_row = i * linhas_por_thread;
        
    
        if (i == (NUM_THREADS - 1)) {
            thread_args[i].end_row = matrixA->height;
        } else {
            thread_args[i].end_row = (i + 1) * linhas_por_thread;
        }
        
        pthread_create(&threads[i], NULL, matrix_mult_worker, &thread_args[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 1;
}

void* matrix_mult_worker(void* args) {
    thread_data* t_args = (thread_data*)args;

    struct matrix *matrixA = t_args->matrixA;
    struct matrix *matrixB = t_args->matrixB;
    struct matrix *matrixC = t_args->matrixC;

  
    for (unsigned long int i = t_args->start_row; i < t_args->end_row; i++) {
        for (unsigned long int k = 0; k < matrixA->width; k++) {

            __m256 a_val_vec = _mm256_set1_ps(matrixA->rows[i * matrixA->width + k]);
            
            unsigned long int j = 0;
            for (j = 0; j <= matrixB->width - 8; j += 8) {
            
                __m256 b_row_vec = _mm256_loadu_ps(&matrixB->rows[k * matrixB->width + j]);
                __m256 c_row_vec = _mm256_loadu_ps(&matrixC->rows[i * matrixC->width + j]);

                // ( A * B ) + C
                __m256 result_vec = _mm256_fmadd_ps(a_val_vec, b_row_vec, c_row_vec);
                
                _mm256_storeu_ps(&matrixC->rows[i * matrixC->width + j], result_vec);
            
            }

            for (; j < matrixB->width; j++) {
                 matrixC->rows[i * matrixC->width + j] += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
            }
        }
    }

    return NULL;
}

void set_number_threads(int num_threads){
    if (num_threads > 0) {
        NUM_THREADS = num_threads;
    }
    else {
        NUM_THREADS = 1;
    }
}