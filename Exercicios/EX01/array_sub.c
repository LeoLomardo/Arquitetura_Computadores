#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "timer.h"
#include <immintrin.h>

int main(int argc, char *argv[]) {
  struct timeval overall_t1, overall_t2;
  long unsigned int N = 1<<21;
  long unsigned int i = 0;
  char *eptr = NULL;

  // Mark overall start time
  gettimeofday(&overall_t1, NULL);

  // Check arguments
  if (argc != 2) {
        printf("Usage: %s <vector_length>\n", argv[0]);
        return 0;
  } else {
        printf("Number of args: %d\n", argc);
        for (i=0; i<argc; ++i)
          printf("argv[%d] = %s\n", i, argv[i]);
  }

  // Convert arguments
  N = strtol(argv[1], &eptr, 10);

  if (N == 0) {
        printf("%s: erro na conversao do argumento: errno = %d\n", argv[0], errno);

        /* If a conversion error occurred, display a message and exit */
        if (errno == EINVAL)
        {
            printf("Conversion error occurred: %d\n", errno);
            return 1;
        }

        /* If the value provided was out of range, display a warning message */
        if (errno == ERANGE) {
            printf("The value provided was out of rangei: %d\n", errno);
            return 1;
	}
  }

  /* Allocate memory for the three arrays */
  float *evens =  (float*)aligned_alloc(32, N*sizeof(float));
  float *odds = (float*)aligned_alloc(32, N*sizeof(float));
  float *result = (float*)aligned_alloc(32, N*sizeof(float));

  if ((evens == NULL) || (odds == NULL) || (result == NULL)) {
	printf("%s: vector allocation problem.", argv[0]);
	return 1;
  }

  /* Initialize the two array in memory */
  float scalar_evens = 8.0f;
  float scalar_odds = 5.0f;
  float *nxt_evens = evens; 
  float *nxt_odds = odds; 

  for ( i = 0; i < N; i += 1, nxt_evens += 1, nxt_odds += 1) {

	  /* Store the elements of the scalar on arrays */
	  *nxt_evens = scalar_evens;
	  *nxt_odds = scalar_odds;
  }

  /* Compute the difference between the two arrays */
  nxt_evens = evens; 
  nxt_odds = odds; 
  float *nxt_result = result; 
  
  /**
  aloquei vetores odds e evens utilizando o _mm256_load_ps()
  alterei o tamanho do i, ja que ele le de 8 em 8 bytes
  utilizei o _mm256_sub_ps() para realizar a subtracao dos valores
  e armazenei o valor do resultado no vetor result utilizando o _mm256_store_ps()
   */
  for ( i = 0; i < N;  i += 8, nxt_evens += 1, nxt_odds += 1, nxt_result += 1) {

    __m256 v1 =  _mm256_load_ps(&evens[i]); 
    __m256 v2 =  _mm256_load_ps(&odds[i]);
    __m256 v3 = _mm256_sub_ps(v1, v2);
    _mm256_store_ps(&result[i], v3);

  }

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  float diffError = 0.0f;
  for (i = 0; i < N; i++)
    maxError = (maxError > (diffError=fabs(result[i]-3.0f)))? maxError : diffError;
  printf("Max error: %f\n", maxError);

  //Mark overall stop time
  gettimeofday(&overall_t2, NULL);
  // Show elased time
  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

  return 0;
}
