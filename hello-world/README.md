# Hello World Parallel Samples

This folder gathers small "hello world" programs that introduce the basic tooling for parallel programming in C, C++, and Fortran. Each subdirectory highlights one parallel programming model so you can compare how the same message-passing idea looks across APIs.

## Directory Layout
```
hello-world/
├── hello-world.c
├── hello-world.cpp
├── hello-world.f
├── mpi/
│   ├── hello-mpi.c
│   └── mpi.sh
├── openmp/
│   └── hello-openmp.c
└── pthreads/
    └── hello-pthreads.c
```

## Top-Level Programs
- `hello-world.c` — Minimal serial C program printing "Hello, World!"; use it as the baseline for parallel versions.
- `hello-world.cpp` — C++ variant using `std::cout`.
- `hello-world.f` — Fortran version for teams working in HPC-centric languages.

## Parallel Samples
- **`mpi/`**
  - `hello-mpi.c` — Uses MPI to run across multiple processes and reports each rank and host.
  - `mpi.sh` — Sample Slurm batch script that loads OpenMPI and launches the compiled MPI program with `srun`.
- **`openmp/`**
  - `hello-openmp.c` — Spawns a team of OpenMP threads (hard-coded to four) and prints each thread ID.
- **`pthreads/`**
  - `hello-pthreads.c` — Demonstrates manual thread creation with POSIX threads and prints from each worker.

## Usage Notes
- Build each sample with the compiler and flags required by your environment (e.g., `gcc hello-world.c -o hello-world`, `mpicc hello-mpi.c -o hello_mpi`, `gcc -fopenmp hello-openmp.c -o hello_openmp`, `gcc hello-pthreads.c -lpthread -o hello_pthreads`).
- The `mpi.sh` script assumes a Slurm-managed cluster; adjust node, task, and module settings to match your system before submitting with `sbatch`.

Use these examples as gentle starting points before scaling up to more complex parallel workloads.
