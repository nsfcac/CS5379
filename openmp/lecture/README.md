# OpenMP Lecture Demos

This folder contains the short programs showcased in lecture to illustrate core
OpenMP concepts. Each source file builds with the provided `Makefile` and runs
on any compiler that supports OpenMP 4.5 or newer (e.g., GCC, Clang, Intel
oneAPI).

## Build

```bash
make            # builds demo-fibonacci.exe, demo-pi.exe, demo-vector.exe
make clean      # removes generated binaries
```

Override the compiler or flags by editing `Makefile` if you want to experiment
with alternatives.

## Demos

- `demo-fibonacci.c` demonstrates nested parallelism with OpenMP tasks. A single
  thread launches two recursive tasks for every call to `fibonacci(n)` and emits
  per-thread diagnostics. The entry point computes `fibonacci(4)` by default;
  adjust `n` in `main` to explore deeper task trees (be mindful that the work
  grows exponentially).

- `demo-pi.c` approximates π via the midpoint rule. The loop over integration
  steps is parallelized with `#pragma omp for reduction(+:sum)`. Pass an
  optional positive integer argument to change the step count, and set
  `OMP_NUM_THREADS` to control the number of workers. The program reports the
  approximation error, elapsed time, and detected thread count.

- `demo-vector.c` highlights data parallelism and memory pressure. It allocates
  very large vectors (hundreds of millions of doubles), initializes them in a
  parallel loop, and then performs a second parallel loop to add `B` and `C`
  into `A`. Reduce the `dimension` constant near the top of the file before
  running on machines with limited memory.

## Suggested Experiments

- Compare serial (`OMP_NUM_THREADS=1`) and parallel runs to observe scalability.
- Enable `OMP_DISPLAY_ENV=true` to have the runtime describe how it configured
  the team.
- Toggle dynamic teams (`export OMP_DYNAMIC=true/false`) and watch the effect on
  `demo-pi`.
- Use `time` or `perf stat` around the executables to capture additional timing
  and hardware-counter information for discussion.
