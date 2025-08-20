#include "matrix_lib.h"
#include <stdlib.h>

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }
    
    unsigned long int total_elements = matrix->height * matrix->width;
    for (int i = 0; i < total_elements; i++) {
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

    for (int i = 0; i < matrixA->height; i++) {
        for (int j = 0; j < matrixB->width; j++) {
            float sum = 0.0;
            for (int k = 0; k < matrixA->width; k++) {
                sum += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
            }
            matrixC->rows[i * matrixC->width + j] = sum;
        }
    }
    
    return 1;
}