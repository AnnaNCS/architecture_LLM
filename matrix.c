#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int matrixSize = 2048;

static inline uint64_t rdtsc() {
  unsigned long a, d;
  asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | ((uint64_t)d << 32);
}

void basicMatMul(int *matA, int *matB, int*result) {
  for (int i = 0; i < matrixSize; i++){
    for (int j = 0; j < matrixSize; j++){
      result[(matrixSize * i) + j] = 0;
      for (int k = 0; k < matrixSize; k++){
        result[(matrixSize * i) + j] += matA[(matrixSize * i) + k] * matB[(matrixSize * k) + j];
      }   
    }   
  }
}

int main() {

  time_t t;
  uint64_t start, end, clock;
  srand((unsigned) time(&t));

  //Data Allocation for basic Matrix Multiplication
  int* matA = malloc(matrixSize * matrixSize * sizeof(int));
  int* matB = malloc(matrixSize * matrixSize * sizeof(int));
  int *result1 = malloc(matrixSize * matrixSize * sizeof(int));
  
  for (int i = 0; i < matrixSize; i++){
    for (int j = 0; j < matrixSize; j++){
      matA[(matrixSize * i) + j] = rand() % 30;
      matB[(matrixSize * i) + j] = rand() % 30;
      result1[(matrixSize * i) + j] = 0;
    }
  }
    
  start = rdtsc();
  basicMatMul(matA, matB, result1);
  end = rdtsc();

  clock = end - start;
  free(matA);
  free(matB);
  free(result1);

  printf("took %lu ticks to find the basic matrix product\n", clock);
  
  return 0;
}