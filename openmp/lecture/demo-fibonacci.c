#include <stdio.h>
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

int main(void) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            int n = 4;
            printf("Fibonacci of %d is %d\n", n, fibonacci(n));
        }
    }

    return 0;
}
