#include <stdint.h> 
#include <string.h> 
#include <unistd.h>
#include <immintrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SZ 2048
#define BLOCK_SIZE 16

static inline uint64_t rdtsc() {
	unsigned long a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return a | ((uint64_t)d << 32);
}

// Function to generate a random value between 0 and 1
double random_double() {
    return ((double)rand() / RAND_MAX);
}

// Basic matrix multiplication with optimization for memory access
void matrix_multiply(double **A, double **B, double **C, int sz) {
    int i, j, k;
    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < sz; i++) {
        for (j = 0; j < sz; j++) {
            // Unroll loop by a factor of 4
            for (k = 0; k < sz; k += 4) {
                C[i][j] += A[i][k] * B[k][j];
                C[i][j] += A[i][k + 1] * B[k + 1][j];
                C[i][j] += A[i][k + 2] * B[k + 2][j];
                C[i][j] += A[i][k + 3] * B[k + 3][j];
            }
        }
    }
}

// Function to allocate memory for a 2D matrix
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

// Function to free memory allocated for a 2D matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(void) {
	// Set seed for random number generation
    srand(time(NULL));

    // Typical size for LLMs
    int sz = 2048;

    // Allocate memory for matrices A, B, and C
    double** A = allocate_matrix(sz, sz);
    double** B = allocate_matrix(sz, sz);
    double** C = allocate_matrix(sz, sz);

    // Initialize matrix A with random values
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            A[i][j] = random_double();
			B[i][j] = random_double();
			C[i][j] = 0.0;
        }
    }

    uint64_t start, end;
    // start count
    _mm_mfence();
    start = rdtsc();

    // Basic matrix multiplication
    matrix_multiply(A, B, C, sz);

	// Perform matrix multiplication using MKL
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sz, sz, sz, 1.0, A, sz, B, sz, 0.0, C, sz);

	// stop count
    end = rdtsc();
    _mm_mfence();

    printf("%lu ticks.\n" , ( end - start));

	// Free allocated memory
    free_matrix(A, sz);
    free_matrix(B, sz);
    free_matrix(C, sz);

    return 0;
}