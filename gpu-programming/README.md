# GPU Programming Lecture - CS5379 Parallel Processing

**Guest Lecture by Mert Side for Dr. Yong Chen**  
**Department of Computer Science, Texas Tech University**  
**September 25th, 2025**

## 📚 Overview

This directory contains comprehensive GPU programming materials for CS5379 Parallel Processing, including educational lecture content and hands-on project assignments. The materials progress from fundamental CUDA concepts to advanced optimization techniques, providing both theoretical understanding and practical implementation experience.

## 📁 Directory Structure

```
gpu-programming/
├── README.md                 # This overview and navigation guide
├── lecture/                  # Complete GPU programming tutorial
│   ├── README.md             # Detailed lecture documentation
│   ├── add_v0.cpp            # CPU baseline implementation
│   ├── add_v1.cu - add_v5.cu # Progressive CUDA optimizations
│   ├── Makefile              # Build system for all versions
│   ├── *.sh                  # Analysis and testing scripts
│   └── cuda_error_check.h    # CUDA error handling utilities
└── project/                  # Individual project assignments
    ├── README.md             # Project-specific instructions
    └── [project files]       # Starter files
```

## 🎯 Learning Path

### **Step 1: Foundational Learning (`lecture/`)**
**Objective**: Master GPU programming fundamentals through progressive optimization.

1. **Start Here**: Read [`lecture/README.md`](lecture/README.md) for complete tutorial
2. **Hands-On Practice**: Use interactive tools to run and analyze all versions
3. **Performance Analysis**: Understand the 21x speedup progression from CPU to optimized GPU

**Key Learning Outcomes**:
- CUDA programming syntax and execution model
- Thread and block parallelization strategies  
- Memory management and optimization techniques
- Performance measurement and profiling skills
- Hardware-aware programming concepts

### **Step 2: Programming Project (`project/`)**
**Objective**: Apply learned concepts to solve computational problems. Please follow the instructions on the PP2 document shared with you on RaiderCanvas.

1. **Project Assignment**: Implement computationally intensive algorithms using CUDA
2. **Optimization Challenge**: Apply lecture concepts to achieve good performance
3. **Analysis & Documentation**: Provide detailed performance analysis and insights

## 🚀 Quick Start Guide

### **Getting Started**

#### **1. Explore the Lecture Materials**
```bash
cd lecture/
# Read the comprehensive tutorial
cat README.md

# Run interactive analysis (requires GPU access)
srun --partition=h100 --gpus-per-node=1 --pty bash
./interactive_test.sh
```

#### **2. Understand the Progression**
```bash
# Compile and run all versions
make all

# Compare performance manually
time ./add_v0.exe  # CPU baseline (~140s)
time ./add_v5.exe  # Optimized GPU (~6.5s - 21x speedup!)
```

#### **3. Move to Project Work**
```bash
cd ../project/
# Follow project-specific instructions
cat README.md
```

## 📊 Performance Observations

### **Lecture Materials Performance Progression**
*Measured on NVIDIA H100 GPU with 1+ billion elements*

| Version | Implementation | Runtime | Speedup | Key Concept |
|---------|----------------|---------|---------|-------------|
| **v0** | CPU Sequential | ~140s | 1.0x | Baseline |
| **v1** | Single GPU Thread | ~120s | 1.2x | CUDA Introduction |
| **v2** | Single Block (256 threads) | ~12s | **11.7x** | Thread Parallelism |
| **v3** | Multiple Blocks | ~11s | **12.7x** | Full GPU Utilization |
| **v4** | GPU Initialization | ~7s | **20.0x** | Memory Optimization |
| **v5** | Memory Prefetching | ~6.5s | **21.5x** | Peak Performance |

### **Key Performance Insights**
- **GPU isn't automatically faster**: v1 shows code without proper parallelization
- **Thread parallelism crucial**: v2 achieves first major speedup (10x+)
- **Hardware utilization matters**: v3 scales across all GPU cores
- **Memory optimization critical**: v4-v5 show significant additional gains
- **Progressive optimization**: Each step builds on previous learnings

## 🛠️ Technical Requirements

### **Hardware Prerequisites**
- **GPU Access**: NVIDIA GPU with CUDA capability (H100 recommended)
- **Memory**: Sufficient for large array processing (8+ GB)

### **Software Dependencies**
- **CUDA Toolkit**: Compatible CUDA compiler (nvcc)
- **GCC Compiler**: Version 14.2.0 or compatible
- **Build Tools**: Make, standard development tools
- **Analysis Tools**: Nsight Systems for profiling (optional)

### **Environment Setup**
```bash
# Load required modules
module load gcc/14.2.0 cuda

# Verify setup
nvcc --version
nvidia-smi
```

**Welcome to the exciting world of GPU programming - where massive parallelism meets computational challenges!** 🚀