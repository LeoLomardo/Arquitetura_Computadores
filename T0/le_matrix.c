#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // Verifica se o n�mero de argumentos est� correto
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <altura> <largura> <arquivo.dat>\n", argv[0]);
        return 1;
    }

    // Converte os argumentos de string para n�meros
    char *eptr;
    unsigned long int height = strtoul(argv[1], &eptr, 10);
    unsigned long int width = strtoul(argv[2], &eptr, 10);
    const char *filename = argv[3];

    unsigned long int total_elements = height * width;

    // Aloca mem�ria para armazenar os floats lidos do arquivo
    float *matrix_data = (float *)malloc(total_elements * sizeof(float));
    if (matrix_data == NULL) {
        fprintf(stderr, "Erro: Falha na aloca��o de mem�ria.\n");
        return 1;
    }

    // Abre o arquivo para leitura em modo bin�rio
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Erro: N�o foi poss�vel abrir o arquivo '%s'.\n", filename);
        free(matrix_data);
        return 1;
    }

    // L� os dados do arquivo para o buffer
    size_t elements_read = fread(matrix_data, sizeof(float), total_elements, file);
    if (elements_read != total_elements) {
        fprintf(stderr, "Aviso: O arquivo '%s' continha menos elementos do que o esperado. Lendo %zu de %lu.\n", filename, elements_read, total_elements);
    }
    
    fclose(file);

    // Imprime a matriz formatada no terminal
    printf("Exibindo matriz do arquivo: %s (%lu x %lu)\n", filename, height, width);
    printf("--------------------------------------------------\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Acessa o elemento na posi��o (i, j) e imprime com 2 casas decimais
            printf("%8.2f ", matrix_data[i * width + j]);
        }
        // Pula para a pr�xima linha ao final de cada linha da matriz
        printf("\n");
    }
    printf("--------------------------------------------------\n");

    // Libera a mem�ria alocada
    free(matrix_data);

    return 0;
}

