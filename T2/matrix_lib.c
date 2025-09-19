#include "matrix_lib.h"
#include <stdlib.h>
#include <immintrin.h> 

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }
    
    unsigned long int total_elements = matrix->height * matrix->width;

    __m256 scalar_vector = _mm256_set1_ps(total_elements);

    for(int i = 0; i < total_elements; i += 8){
       __m256 matrix_vector = _mm256_loadu_ps(&matrix->rows[i]);
       __m256 result_vector = _mm256_mul_ps(matrix_vector, scalar_vector);
       _mm256_storeu_ps(&matrix->rows[i], result_vector);
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
    
    
    for (unsigned int i = 0; i < matrixA->height; i++) {
        for (unsigned int k = 0; k < matrixA->width; k++) {

            __m256 a_linhaToda = _mm256_set1_ps(matrixA->rows[i * matrixA->width + k]);

            for (unsigned int j = 0; j < matrixB->width; j += 8) {
    
                __m256 b_vector = _mm256_loadu_ps(&matrixB->rows[k * matrixB->width + j]);
                
                __m256 c_vector = _mm256_loadu_ps(&matrixC->rows[i * matrixC->width + j]);
                
                //Como mostrado no slide:: A = A + (B x C)
                c_vector = _mm256_fmadd_ps(a_linhaToda, b_vector, c_vector);
                
                _mm256_storeu_ps(&matrixC->rows[i * matrixC->width + j], c_vector);
            }
        }
    }
    
    return 1;
}
