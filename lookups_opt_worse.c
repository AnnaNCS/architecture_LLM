#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h> 
#include <omp.h>

#define EMBEDDING_SIZE 1024
#define VOCAB_SIZE 50000

static inline uint64_t rdtsc() {
    unsigned long a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return a | ((uint64_t)d << 32);
}

void embedding_lookup(double* embedding_matrix, int* indices, double* result, int num_indices) {
    // #pragma omp parallel for
    for (int i = 0; i < num_indices; ++i) {
        int offset = indices[i] * EMBEDDING_SIZE;
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
            result[i * EMBEDDING_SIZE + j] = embedding_matrix[offset + j];
        }
    }
}

int main() {
    // Seed for reproducibility
    // srand(42);

    // Dynamically allocate memory for embedding matrix
    double* embedding_matrix = (double*)malloc(VOCAB_SIZE * EMBEDDING_SIZE * sizeof(double));
    for (int i = 0; i < VOCAB_SIZE * EMBEDDING_SIZE; ++i) {
        embedding_matrix[i] = (double)rand() / RAND_MAX;
    }

    // Dynamically allocate memory for indices
    int num_indices = 20;
    int indices[num_indices];
    for (int i = 0; i < num_indices; ++i) {
        indices[i] = rand() % VOCAB_SIZE;
    }

    // Dynamically allocate memory for embeddings
    double* embeddings = (double*)malloc(num_indices * EMBEDDING_SIZE * sizeof(double));

    uint32_t clock, start, end;
	// Start count
    clock = 0;
    _mm_mfence();
    start = rdtsc();
    
    // Perform embedding lookup
    embedding_lookup(embedding_matrix, indices, embeddings, num_indices);
    
    // Stop count
    end = rdtsc();
    _mm_mfence();
    clock = clock + (end - start);

    // Print number of ticks
    printf("%u ticks.\n" , (end - start));

    // Display the results
    // for (int i = 0; i < num_indices; ++i) {
    //     printf("Embedding for index %d:\n", indices[i]);
    //     for (int j = 0; j < EMBEDDING_SIZE; ++j) {
    //         printf("%f ", embeddings[i * EMBEDDING_SIZE + j]);
    //     }
    //     printf("\n");
    // }

    // Free dynamically allocated memory
    free(embedding_matrix);
    free(embeddings);

    return 0;
}
