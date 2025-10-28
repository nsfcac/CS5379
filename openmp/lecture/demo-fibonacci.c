#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static int fibonacci(int n) {
    int thread_id = omp_get_thread_num();

    if (n < 2) {
        #pragma omp critical
        {
            printf("Thread %d computed fibonacci(%d) = %d\n", thread_id, n, n);
        }
        return n;
    }

    int x = 0;
    int y = 0;
    int result = 0;

    #pragma omp task shared(x)
    {
        x = fibonacci(n - 1);
    }

    #pragma omp task shared(y)
    {
        y = fibonacci(n - 2);
    }

    #pragma omp taskwait
    result = x + y;

    #pragma omp critical
    {
        printf("Thread %d computed fibonacci(%d) = %d\n", thread_id, n, result);
    }

    return result;
}

int main(int argc, char **argv) {
    int n = 4;

    if (argc > 1) {
        char *end = NULL;
        long value = strtol(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0') {
            fprintf(stderr, "Invalid length '%s'. Using default %d.\n", argv[1], n);
        } else if (value < 0) {
            fprintf(stderr, "Negative length '%s'. Using default %d.\n", argv[1], n);
        } else {
            n = (int)value;
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Fibonacci of %d is %d\n", n, fibonacci(n));
        }
    }

    return 0;
}
