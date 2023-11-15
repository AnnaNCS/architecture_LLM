#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void softmax(double *x, int length) {
    double max_val = x[0];
    
    // Find the maximum value in the array
    for (int i = 1; i < length; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // Calculate the exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    // Normalize by the sum
    for (int i = 0; i < length; i++) {
        x[i] /= sum;
    }
}

int main() {
    // Typical vector length for LLM vocabulary
    int length = 50000;
    
    // Allocate memory for the array
    double
