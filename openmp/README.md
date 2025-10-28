# OpenMP Materials

This directory gathers the OpenMP resources used in CS5379, split into lecture
demos and the semester programming project.

## Lecture demos

The `lecture/` folder now contains every demo shown during lecture, including
`demo-fibonacci.c`, `demo-pi.c`, and `demo-vector.c`. Build them by running
`make` inside `openmp/lecture`; the provided `Makefile` emits the corresponding
`.exe` binaries. Clean artifacts with `make clean`.

To explore performance, set `OMP_NUM_THREADS` before launching any demo. For
example:

```bash
cd openmp/lecture
make
./demo-pi.exe 5000000
OMP_NUM_THREADS=8 ./demo-vector.exe
```

Each program prints its detected thread count and timing information so you can
compare scaling across different workloads and thread configurations.

## Programming project

The `project/` folder holds the OpenMP programming project sources, currently
`omp-gaussian-elimination.c` and `omp-matrix-mul.c`. Refer to the programming project 3 for more details. 
