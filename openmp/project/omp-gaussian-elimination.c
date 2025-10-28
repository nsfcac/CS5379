#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double *A;
    double *b;
    size_t n;
} LinearSystem;

static void initialize_system(LinearSystem *system) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    size_t n = system->n;
    double *truth = (double *)malloc(n * sizeof(double));
    if (!truth) {
        fprintf(stderr, "Failed to allocate buffer for synthetic solution\n");
        exit(EXIT_FAILURE);
    }

    for (size_t j = 0; j < n; ++j) {
        int value = (rand() % 25) - 12; /* range [-12, 12] */
        if (value == 0) {
            value = 7;
        }
        truth[j] = (double)value;
    }

    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            int value = (rand() % 21) - 10; /* range [-10, 10] */
            if (value == 0) {
                value = -3;
            }
            system->A[i * n + j] = (double)value;
            row_sum += (value < 0) ? -(double)value : (double)value;
        }

        int extra = (rand() % 5) + 1;
        double diag = row_sum + (double)(extra + (int)n);
        system->A[i * n + i] = diag;

        double rhs = 0.0;
        for (size_t j = 0; j < n; ++j) {
            rhs += system->A[i * n + j] * truth[j];
        }
        system->b[i] = rhs;
    }

    free(truth);
}

static int gaussian_elimination(double *A, double *b, size_t n) {
    for (size_t k = 0; k + 1 < n; ++k) {
        double diag = A[k * n + k];
        for (size_t i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / diag;
            A[i * n + k] = 0.0;
            for (size_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    return 0;
}

static int back_substitution(const double *A, const double *b, double *x, size_t n) {
    for (ptrdiff_t row = (ptrdiff_t)n - 1; row >= 0; --row) {
        double rhs = b[row];
        for (size_t col = (size_t)row + 1; col < n; ++col) {
            rhs -= A[row * n + col] * x[col];
        }
        double diag = A[row * n + (size_t)row];
        x[row] = rhs / diag;
    }
    return 0;
}

static void print_vector(const char *label, const double *v, size_t n) {
    printf("%s", label);
    for (size_t i = 0; i < n; ++i) {
        printf(" % .6f", v[i]);
    }
    putchar('\n');
}

int main(int argc, char *argv[]) {
    size_t n = 4;
    if (argc > 1) {
        char *endptr = NULL;
        long parsed = strtol(argv[1], &endptr, 10);
        if (endptr == argv[1] || parsed <= 1) {
            fprintf(stderr, "Provide a matrix dimension greater than 1.\n");
            return EXIT_FAILURE;
        }
        n = (size_t)parsed;
    }

    LinearSystem system = {
        .A = (double *)malloc(n * n * sizeof(double)),
        .b = (double *)malloc(n * sizeof(double)),
        .n = n,
    };
    double *x = (double *)calloc(n, sizeof(double));

    if (!system.A || !system.b || !x) {
        fprintf(stderr, "Failed to allocate memory for %zux%zu system.\n", n, n);
        free(system.A);
        free(system.b);
        free(x);
        return EXIT_FAILURE;
    }

    initialize_system(&system);

    int status = gaussian_elimination(system.A, system.b, n);
    if (status == 0) {
        status = back_substitution(system.A, system.b, x, n);
    }

    if (status == 0) {
        print_vector("Solution:", x, n);
    }

    free(system.A);
    free(system.b);
    free(x);
    return (status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
