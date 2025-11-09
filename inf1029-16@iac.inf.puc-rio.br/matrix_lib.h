#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *h_rows;
    float *d_rows;
    int alloc_mode;
};

typedef struct {
    struct matrix *matrixA;
    struct matrix *matrixB;
    struct matrix *matrixC;
    unsigned long int start_row; 
    unsigned long int end_row;   
} thread_data;

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);
int set_grid_size(int threads_per_block, int max_blocks_per_grid);
void* matrix_mult_worker(void* args);
void set_number_threads(int num_threads);

#endif