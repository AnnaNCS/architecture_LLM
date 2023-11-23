#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdint.h> 
#include <string.h> 
#include <unistd.h>
#include <immintrin.h>

static inline uint64_t rdtsc() {
    unsigned long a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return a | ((uint64_t)d << 32);
}

void softmax(float *x, int length) {
    
    float max_val = x[0];
    
    // Find the maximum value in the array
    for (int i = 1; i < length; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // 1.6

    // Compute Softmax in parallel
	for (int i = 0; i < length; i++) {
		x[i] = exp(x[i] - max_val);
	}
    // 0.8

    // Calculate the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += x[i];
    }
    // 1.5

    // Normalize by the sum
    for (int i = 0; i < length; i++) {
        x[i] /= sum;
    }
    // 1.6
}

int main() {
    // Typical vector length for LLM vocabulary
    int length = 50000;
    uint32_t result_vect;
    uint32_t clock, start, end;
    
    // Allocate memory for the array
    float *x = (float *)malloc(length * sizeof(float));

    // Populate the array with random values
    for (int i = 0; i < length; i++) {
        x[i] = ((float)rand() / RAND_MAX);
    }

    // Start count
    clock = 0;
    _mm_mfence();
    start = rdtsc();
    
    // Softmax computation
    softmax(x, length);
    
    // Stop count
    end = rdtsc();
    _mm_mfence();
    clock = clock + (end - start);

    // Print number of ticks
    printf("%u ticks.\n" , (end - start));

    // Print the result
    // printf("Softmax Result:\n");
    // for (int i = 0; i < length; i++) {
        // printf("%f ", x[i]);
    // }
    
    // Free allocated memory
    free(x);

    return 0;
}
