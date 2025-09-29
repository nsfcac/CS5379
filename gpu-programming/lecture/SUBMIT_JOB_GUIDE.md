# SLURM Job Submission Guide: `submit_job.sh`

This guide explains how to use the `submit_job.sh` script for running CUDA vector addition programs on the HPC cluster.

## Overview

The `submit_job.sh` script is a SLURM batch script that automatically compiles and runs different versions of vector addition programs, ranging from CPU-only to optimized CUDA implementations. It provides performance timing and profiling capabilities.

## Prerequisites

- Access to an HPC cluster with SLURM workload manager
- H100 GPU partition availability
- GCC 14.2.0 and CUDA modules

## Usage

### Basic Syntax

```bash
sbatch submit_job.sh [version]
```

### Available Versions

| Version | Description | File | Compiler |
|---------|-------------|------|----------|
| `v0` | CPU baseline (C++) | `add_v0.cpp` | `g++` |
| `v1` | Basic CUDA | `add_v1.cu` | `nvcc` |
| `v2` | CUDA with optimizations | `add_v2.cu` | `nvcc` |
| `v3` | Advanced CUDA features | `add_v3.cu` | `nvcc` |
| `v4` | Further optimizations | `add_v4.cu` | `nvcc` |
| `v5` | Most optimized (default) | `add_v5.cu` | `nvcc` |

## Examples

### Run Default Version (v5)
```bash
sbatch submit_job.sh
```

### Run Specific Version
```bash
# Run CPU baseline
sbatch submit_job.sh v0

# Run basic CUDA version
sbatch submit_job.sh v1

# Run optimized version
sbatch submit_job.sh v3
```

### Check Job Status
```bash
# View queue status
squeue -u $USER

# Monitor specific job
squeue -j <job_id>

# Cancel job if needed
scancel <job_id>
```

## Resource Allocation

The script requests the following resources:
- **Partition**: `h100` (H100 GPU nodes)
- **Nodes**: 1
- **Tasks per node**: 32
- **GPUs per node**: 1
- **Job name**: `CS5379_eraider_ID`

## Output Files

### Standard Output/Error Files
- **Pattern**: `CS5379_eraider_ID_<job_id>_<version>.out/.err`
- **Examples**: 
  - `CS5379_eraider_ID_12345_v3.out`
  - `CS5379_eraider_ID_12345_v3.err`

### Profiling Files (CUDA versions only)
- **Nsight Systems**: `*.nsys-rep`
- **SQLite databases**: `*.sqlite`

## What the Script Does

### 1. Validation
- Validates the version argument
- Loads required modules (GCC 14.2.0, CUDA)

### 2. Compilation
- **CPU version (v0)**: Uses `g++` to compile `.cpp` file
- **CUDA versions (v1-v5)**: Uses `nvcc` to compile `.cu` files

### 3. Execution & Timing

#### CPU Version (v0)
- Runs 3 iterations for statistical accuracy
- Provides baseline performance metrics
- No GPU profiling

#### CUDA Versions (v1-v5)
- Single timed execution
- GPU performance metrics
- Nsight Systems profiling

### 4. Profiling (CUDA only)
- Generates performance profiles using `nsys nvprof`
- Creates `.nsys-rep` and `.sqlite` files for analysis

## Performance Analysis

### CPU Baseline (v0)
```
🖥️  CPU Baseline Performance (C++ version):
Running 3 iterations for average timing...

Run 1/3:
  Execution time: 0m2.345s
📊 CPU Performance Summary:
  - Pure CPU execution (single-threaded)
  - No GPU acceleration
  - Baseline for comparison
```

### CUDA Versions (v1-v5)
```
🚀 CUDA Performance (Version v3):
Running timed execution:
[program output with timing]

📊 Profiling with Nsight Systems...
✅ Profiling completed
```

## Troubleshooting

### Common Issues

1. **Invalid version error**
   ```
   ❌ Invalid version: v6
   Valid versions: v0, v1, v2, v3, v4, v5
   ```
   **Solution**: Use a valid version number

2. **Compilation failed**
   ```
   ❌ Compilation failed
   Check error messages above
   ```
   **Solution**: Check source files exist and syntax is correct

3. **Job stuck in queue**
   ```bash
   # Check partition availability
   sinfo -p h100
   
   # Check your job details
   scontrol show job <job_id>
   ```

### File Permissions
Ensure the script is executable:
```bash
chmod +x submit_job.sh
```

## Performance Comparison Workflow

1. **Start with CPU baseline**:
   ```bash
   sbatch submit_job.sh v0
   ```

2. **Test basic CUDA**:
   ```bash
   sbatch submit_job.sh v1
   ```

3. **Compare optimized versions**:
   ```bash
   sbatch submit_job.sh v3
   sbatch submit_job.sh v5
   ```

4. **Analyze results**:
   - Compare timing output in `.out` files
   - Use Nsight Systems to analyze `.nsys-rep` files

## Tips for Best Results

- **Sequential testing**: Submit jobs one at a time to avoid resource conflicts
- **Version comparison**: Always include v0 (CPU) for baseline comparison
- **Profile analysis**: Use Nsight Systems GUI to analyze profiling data
- **Resource monitoring**: Check cluster load before submitting multiple jobs

## Module Dependencies

The script automatically loads:
```bash
module load gcc/14.2.0 cuda
```

Ensure these modules are available on your system.

---

**Note**: Replace `eraider_ID` in the job name with your actual eRaider ID for proper job identification.