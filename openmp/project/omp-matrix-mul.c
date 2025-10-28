#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void initialize_matrix(double *matrix, size_t n, double seed_factor) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i * n + j] = seed_factor * (double)(i + j);
        }
    }
}

int main(int argc, char *argv[]) {
    size_t n = 2048; /* default dimension large enough to see non-zero wall time */

    if (argc > 1) {
        char *endptr = NULL;
        long parsed = strtol(argv[1], &endptr, 10);
        if (endptr == argv[1] || parsed <= 0) {
            fprintf(stderr, "Invalid matrix dimension: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        n = (size_t)parsed;
    }

    size_t total_elements = n * n;
    double *A = (double *)malloc(total_elements * sizeof(double));
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C = (double *)calloc(total_elements, sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate memory for %zu x %zu matrices\n", n, n);
        free(A);
        free(B);
        free(C);
        return EXIT_FAILURE;
    }

    initialize_matrix(A, n, 1.0);
    initialize_matrix(B, n, 0.5);

    double start = omp_get_wtime();
    int used_threads = 1;

    // Tips: use the following code to update the used_threads variable in the parallel region
    // #pragma omp single
    // {
    //     used_threads = omp_get_num_threads();
    // }

    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("Computed %zux%zu matrix multiply in %.3f seconds using %d threads\n",
           n, n, end - start, used_threads);

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
