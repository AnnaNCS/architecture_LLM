#include <stdio.h>

#define N 2048

//MATRIX MULT CODE 
void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Initialize matrices A, B, and C
    int A[N][N], B[N][N], C[N][N];

    // Assume matrices A and B are filled with appropriate values

    // Perform matrix multiplication
    matrix_multiply(A, B, C);

    // Print the result matrix C if needed
    /*
    printf("Resultant Matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
    */

    return 0;
}