double my_exp(double x) {
    const int terms = 20;
    double result = 1.0;
    double term = 1.0;

    for (int i = 1; i < terms; ++i) {
        term *= x / i;
        result += term;
    }

    return result;
}

void vectorized_exp(float* input, float* output, int length) {
    // Process four elements at a time using AVX
    for (int i = 0; i < length; i += 8) {
        // Load 8 floats from input into AVX register
        __m256 input_vector = _mm256_loadu_ps(&input[i]);

        // Compute exp for each element in the vector
        __m256 exp_vector = _mm256_exp_ps(input_vector);

        // Store the result back to output
        _mm256_storeu_ps(&output[i], exp_vector);
    }
}