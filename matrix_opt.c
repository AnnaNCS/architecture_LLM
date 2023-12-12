#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

int matrixSize = 2048;

static inline uint64_t rdtsc() {
  unsigned long a, d;
  asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | ((uint64_t)d << 32);
}

void CFriendlyMatMul(int *matA, int *matB, int *result) {
  int tile = 16;
  for (int io = 0; io < matrixSize; io = io + tile){
    for (int ko = 0; ko < matrixSize; ko = ko + tile){
      for (int jo = 0; jo < matrixSize; jo = jo + tile){
        for(int i = io; i < io + tile; i++){
          for(int k = ko; k < ko + tile; k++){
            for(int j = jo; j < jo + tile; j++){
              result[(matrixSize * i) + j] += matA[(matrixSize * i) + k] * matB[(matrixSize * k) + j]; //no stride access + loop blocking
            }
          }
        }
      }      
    }
  }
}

int main() {

  time_t t;
  uint64_t start, end, clock;
  srand((unsigned) time(&t));
  

  //Data Allocation for Cache Friendly Matrix Multiplication
  int* matA = malloc(matrixSize * matrixSize * sizeof(int));
  int* matB = malloc(matrixSize * matrixSize * sizeof(int));
  int *result2 = malloc(matrixSize * matrixSize * sizeof(int));
  
  for (int i = 0; i < matrixSize; i++){
    for (int j = 0; j < matrixSize; j++){
      matA[(matrixSize * i) + j] = rand() % 30;
      matB[(matrixSize * i) + j] = rand() % 30;
      result2[(matrixSize * i) + j] = 0;
    }
  }

  start = rdtsc();
  CFriendlyMatMul(matA, matB, result2);
  end = rdtsc();

  clock = end - start;
  free(matA);
  free(matB);
  free(result2);

  printf("took %lu ticks to find the Cache friendly matrix product\n", clock);
  
  return 0;
}