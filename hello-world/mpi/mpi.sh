#!/bin/bash
#SBATCH --job-name=MPI_TEST_JOB
#SBATCH --partition=zen4
#SBATCH --ntasks=16
#SBATCH --time=00:05:00
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load openmpi/5.0.4

srun ./hello-mpi
