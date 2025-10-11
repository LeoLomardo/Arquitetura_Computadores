#include "matrix_lib.h"
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>
#include <pthread.h>


static int NUM_THREADS = -1; // defini -1 apenas para garantir que a funcao q define o numero de threads seja chamada e esteja funcionando

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }
    
    unsigned long int total_elements = matrix->height * matrix->width;
    __m256 scalar_vector = _mm256_set1_ps(scalar_value);
    
    unsigned long int i = 0;
    for (i = 0; i <= total_elements - 8; i += 8) {
        __m256 matrix_vector = _mm256_loadu_ps(&matrix->rows[i]);
        __m256 result_vector = _mm256_mul_ps(matrix_vector, scalar_vector);
        _mm256_storeu_ps(&matrix->rows[i], result_vector);
    }
    
    for (; i < total_elements; i++) {
        matrix->rows[i] *= scalar_value;
    }
    
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