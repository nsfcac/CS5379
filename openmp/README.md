# OpenMP Lecture Demos

This directory collects the two OpenMP examples demonstrated in lecture.
Each program compiles with the provided `Makefile` and illustrates a distinct
concept of shared-memory parallelism.

- `demo-pi.c` approximates pi with the midpoint rule, using
  `#pragma omp parallel for reduction(+:sum)` so every thread contributes to the
  global integral safely. Pass an optional positive integer argument to change
  the number of integration steps.

- `demo-vector.c` launches threads that initialize and sum very large vectors.
  The `#pragma omp parallel` / `for` combination shows how work-sharing and
  synchronization interact. The default dimension is intentionally enormous
  (~10 GB of doubles) so it will typically exhaust memory; reduce the constant
  at the top of the file when running on smaller machines.

## Prerequisites

- GCC (or another compiler with OpenMP 4.5+ support) and the OpenMP runtime

## Build

```bash
make            # builds both demos as *.exe binaries
```

Clean artifacts with `make clean`.

## Run

Control the thread count with the standard `OMP_NUM_THREADS` environment
variable. Example invocations:

```bash
./demo-pi.exe 5000000
OMP_NUM_THREADS=8 ./demo-vector.exe
```

`demo-pi` reports the elapsed wall-clock time and the thread count detected at
runtime. `demo-vector` prints the number of worker threads and the duration of
the addition kernel.

## Suggested Experiments

- Compare `OMP_NUM_THREADS=1` vs. the physical core count to illustrate scaling.
- Vary the iteration count (`demo-pi`) or the vector dimension (`demo-vector`)
  to explore how workload size influences speedup and memory pressure.
- Inspect the `Makefile` to review the flags required for OpenMP (`-fopenmp`).
