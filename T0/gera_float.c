#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <altura> <largura> <nome_do_arquivo>\n", argv[0]);
        return 1;
    }

    char *eptr;
    unsigned long int height = strtoul(argv[1], &eptr, 10);
    unsigned long int width = strtoul(argv[2], &eptr, 10);
    const char *filename = argv[3];

    unsigned long int total_elements = height * width;

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Erro: Nao foi possivel abrir o arquivo %s para escrita.\n", filename);
        return 1;
    }

    srand(time(NULL));

    for (int i = 0; i < total_elements; i++) {
        float random_float = (float)rand() / (float)RAND_MAX * 10.0f; 
        fwrite(&random_float, sizeof(float), 1, file);
    }

    fclose(file);

    printf("Arquivo '%s' gerado com sucesso contendo %lu floats.\n", filename, total_elements);

    return 0;
}

