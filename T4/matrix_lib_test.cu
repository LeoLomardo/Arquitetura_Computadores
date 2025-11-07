#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"
#include "timer.h"

void allocate_matrix(struct matrix *m) {
    m->h_rows = (float *)malloc(m->height * m->width * sizeof(float));
    if (m->h_rows == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(1);
    }
}

void deallocate_matrix(struct matrix *m) {
    if (m != NULL && m->h_rows != NULL) {
        free(m->h_rows);
    }
}

void load_matrix_from_file(const char *filename, struct matrix *m) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s for reading.\n", filename);
        exit(1);
    }
    size_t elements_to_read = m->height * m->width;
    size_t elements_read = fread(m->h_rows, sizeof(float), elements_to_read, file);
	fclose(file);

    if (elements_read != elements_to_read) {
        fprintf(stderr, "Error: Failed to read the correct number of elements from %s.\n", filename);
        exit(1);
    }
}

void save_matrix_to_file(const char *filename, struct matrix *m) {
    FILE *file = fopen(filename, "wb+");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        exit(1);
    }
    size_t elements_to_write = m->height * m->width;
    size_t elements_written = fwrite(m->h_rows, sizeof(float), elements_to_write, file);
	fclose(file);

    if (elements_written != elements_to_write) {
        fprintf(stderr, "Error: Failed to write the correct number of elements to %s.\n", filename);
        exit(1);
    }
}

void print_matrix_elements(const char* name, struct matrix *m) {
    printf("Matrix %s:\n", name);
    int limit = 256;

    for (int i = 0; i < limit; i++) {
        printf("%f ", m->h_rows[i]);
        if ((i + 1) % 16 == 0) { 
            printf("\n");
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc != 13) {
        fprintf(stderr, "Usage: %s <scalar> <A_height> <A_width> <B_height> <B_width> <NUM_THREADS> <BLOCS_PER_GRID> <MAX_MEMORY> <file_A> <file_B> <file_result1> <file_result2>\n", argv[0]);
        return 1;
    }

    struct timeval overall_t1, overall_t2, start, stop;
    gettimeofday(&overall_t1, NULL);

    float scalar_value = atof(argv[1]);
    
    struct matrix matrixA, matrixB, matrixC;
    
    matrixA.height = strtoul(argv[2], NULL, 10);
    matrixA.width = strtoul(argv[3], NULL, 10);
    matrixB.height = strtoul(argv[4], NULL, 10);
    matrixB.width = strtoul(argv[5], NULL, 10);
    int threads_per_block = strtol(argv[6], NULL, 10);
	int max_blocks = strtol(argv[7], NULL, 10);
	int max_memory = strtol(argv[8], NULL, 10);

    if (matrixA.width != matrixB.height) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication.\n");
        return 1;
    }
    
    matrixC.height = matrixA.height;
    matrixC.width = matrixB.width;

    allocate_matrix(&matrixA);
    allocate_matrix(&matrixB);
    allocate_matrix(&matrixC);

    load_matrix_from_file(argv[9], &matrixA);
    load_matrix_from_file(argv[10], &matrixB);

    gettimeofday(&start, NULL);
    if (!scalar_matrix_mult(scalar_value, &matrixA)) {
        fprintf(stderr, "Error during scalar matrix multiplication.\n");
        return 1;
    }
    gettimeofday(&stop, NULL);
    printf("Scalar-matrix multiplication time: %f ms\n", timedifference_msec(start, stop));

    save_matrix_to_file(argv[11], &matrixA);
    
    // gettimeofday(&start, NULL);
    // if (!matrix_matrix_mult(&matrixA, &matrixB, &matrixC)) {
    //     fprintf(stderr, "Error during matrix matrix multiplication.\n");
    //     return 1;
    // }
    // gettimeofday(&stop, NULL);
    // printf("Matrix-matrix multiplication time: %f ms\n", timedifference_msec(start, stop));

    // save_matrix_to_file(argv[12], &matrixC);

    //imprime as matrizes
    print_matrix_elements("Matriz A:", &matrixA);
    // print_matrix_elements("Matriz B:", &matrixB);
    // print_matrix_elements("Matriz C:", &matrixC);

    deallocate_matrix(&matrixA);
    deallocate_matrix(&matrixB);
    deallocate_matrix(&matrixC);
    
    gettimeofday(&overall_t2, NULL);
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}

