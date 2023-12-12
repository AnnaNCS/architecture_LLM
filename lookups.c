#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h> 

#define EMBEDDING_SIZE 1024
#define VOCAB_SIZE 50000

static inline uint64_t rdtsc() {
    unsigned long a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return a | ((uint64_t)d << 32);
}

// Function to perform embedding lookup
void embedding_lookup(double** embedding_matrix, int* indices, double** result, int num_indices) {
    for (int i = 0; i < num_indices; ++i) {
		for (int j = 0; j < EMBEDDING_SIZE; ++j) {
			result[i][j] = embedding_matrix[indices[i]][j];
        } 
    }
}

int main() {
    // Seed for reproducibility
    srand(42);

    // Dynamically allocate memory for embedding matrix
    double** embedding_matrix = (double**)malloc(VOCAB_SIZE * sizeof(double*));
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        embedding_matrix[i] = (double*)malloc(EMBEDDING_SIZE * sizeof(double));
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
            embedding_matrix[i][j] = (double)rand() / RAND_MAX; // Random values between 0 and 1
        }
    }


    // Dynamically allocate memory for indices
    int num_indices = 20;
    int indices[num_indices];
    for (int i = 0; i < num_indices; ++i) {
        indices[i] = rand() % VOCAB_SIZE;
    }

    // Dynamically allocate memory for embeddings
    double** embeddings = (double**)malloc(num_indices * sizeof(double*));
    for (int i = 0; i < num_indices; ++i) {
        embeddings[i] = (double*)malloc(EMBEDDING_SIZE * sizeof(double));
    }

	uint64_t start, end;
	// Start count
    _mm_mfence();
    start = rdtsc();
    
    // Perform embedding lookup
    embedding_lookup(embedding_matrix, indices, embeddings, num_indices);
    
    // Stop count
    end = rdtsc();
    _mm_mfence();

    // Print number of ticks
    printf("%lu ticks.\n" , (end - start));

    // Display the results
    // for (int i = 0; i < num_indices; ++i) {
    //     printf("Embedding for index %d:\n", indices[i]);
    //     for (int j = 0; j < EMBEDDING_SIZE; ++j) {
    //         printf("%f ", embeddings[i][j]);
    //     }
    //     printf("\n");
    // }

    // Free dynamically allocated memory
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        free(embedding_matrix[i]);
    }
    free(embedding_matrix);

    for (int i = 0; i < num_indices; ++i) {
        free(embeddings[i]);
    }
    free(embeddings);

    return 0;
}
