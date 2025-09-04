#include "matrix_lib.h"
#include <stdlib.h>


int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }
    
    unsigned long int total_elements = matrix->height * matrix->width;
    for (unsigned long int i = 0; i < total_elements; i++) {
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
    
    for (unsigned long int i = 0; i < matrixA->height; i++) {

        for (unsigned long int k = 0; k < matrixA->width; k++) {
            //como mostrado no slide, ele acessa o valor da variavel A apenas 1 vez, otimizando o acesso a memoria 
            float A_i_k = matrixA->rows[i * matrixA->width + k];

            for (unsigned long int j = 0; j < matrixB->width; j++) {

                matrixC->rows[i * matrixC->width + j] += A_i_k * matrixB->rows[k * matrixB->width + j];
            }
        }
    }
    
    return 1;
}