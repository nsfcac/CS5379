#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    const int dimension = 430420230;
    double *A = malloc((size_t)dimension * sizeof(*A));
    double *B = malloc((size_t)dimension * sizeof(*B));
    double *C = malloc((size_t)dimension * sizeof(*C));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Allocation failed. Unable to continue.\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    #pragma omp parallel for
    for (int i = 0; i < dimension; ++i) {
        A[i] = 0.0;
        B[i] = 1.0;
        C[i] = 2.0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Computation starts with %d threads.\n", omp_get_num_threads());
            fflush(stdout);
        }

        #pragma omp for
        for (int i = 0; i < dimension; ++i) {
            A[i] = B[i] + C[i];
        }
    }

    double elapsed = omp_get_wtime() - start;
    printf("Computation ends in %f seconds.\n", elapsed);

    free(A);
    free(B);
    free(C);
    return 0;
}
