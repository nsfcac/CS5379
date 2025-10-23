#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    long steps = 100000000L;

    if (argc > 1) {
        char *end = NULL;
        long value = strtol(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && value > 0) {
            steps = value;
        } else {
            fprintf(stderr, "Invalid step count '%s'. Using default %ld.\n", argv[1], steps);
        }
    }

    double step = 1.0 / (double)steps;
    double sum = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < steps; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum * step;
    double elapsed = omp_get_wtime() - start;

    printf("steps = %ld, pi approx %.12f\n", steps, pi);
    printf("elapsed = %.6f seconds with %d threads\n", elapsed, omp_get_max_threads());

    return 0;
}
