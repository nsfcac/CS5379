# GPU Programming Lecture - CS5379 Parallel Processing

**Guest Lecture by Mert Side for Dr. Yong Chen**  
**Department of Computer Science, Texas Tech University**  
**September 25th, 2025**

## 🎯 Lecture Overview

This comprehensive tutorial demonstrates GPU programming through progressive optimization of vector addition. We start with a CPU baseline and advance through increasingly optimized CUDA implementations, achieving a **21x speedup** while learning fundamental GPU programming concepts.

**Core Learning Objective**: Understand how proper GPU programming requires strategic parallelization, memory management, and hardware-aware optimization.

## 📚 Program Versions & Educational Progression

### **Version 0 (add_v0.cpp) - CPU Baseline** 
- **Purpose**: Establish performance baseline and demonstrate traditional CPU approach
- **Implementation**: Sequential C++ processing with standard memory allocation
- **Key Concepts**: Baseline measurement, CPU memory management, validation techniques
- **Performance**: ~140 seconds (reference point for all comparisons)
- **Learning Focus**: Understanding the starting point before GPU optimization

### **Version 1 (add_v1.cu) - First CUDA Implementation**
- **Purpose**: Demonstrate that GPU isn't automatically faster without proper parallelization
- **Implementation**: Single GPU thread (1×1 grid configuration)
- **Key Concepts**: CUDA syntax, `__global__` kernels, unified memory, synchronization
- **Performance**: ~120 seconds (slower than CPU due to GPU overhead!)
- **Learning Focus**: Common beginner mistake - GPU requires parallel thinking

### **Version 2 (add_v2.cu) - Thread-Level Parallelism**
- **Purpose**: First significant GPU acceleration through thread parallelism
- **Implementation**: Single block with 256 threads (1×256 configuration)
- **Key Concepts**: Thread indexing (`threadIdx.x`), strided memory access, parallel work distribution
- **Performance**: ~12 seconds (**10x+ speedup!**)
- **Learning Focus**: Power of parallel threads working together

### **Version 3 (add_v3.cu) - Full GPU Utilization**
- **Purpose**: Maximize GPU hardware utilization across all Streaming Multiprocessors
- **Implementation**: Multiple blocks with 256 threads each (4M+ × 256 configuration)
- **Key Concepts**: Global thread indexing, grid-stride loops, SM utilization, scalability
- **Performance**: ~11 seconds (scales with GPU hardware capabilities)
- **Learning Focus**: Block-level parallelism and hardware-aware programming

### **Version 4 (add_v4.cu) - Memory Initialization Optimization**
- **Purpose**: Optimize memory operations by moving initialization to GPU
- **Implementation**: Dual-kernel approach (GPU initialization + computation)
- **Key Concepts**: Multiple kernel launches, memory bandwidth optimization, GPU-based initialization
- **Performance**: ~7 seconds (**35% improvement from v3!**)
- **Learning Focus**: Memory management impact on overall performance

### **Version 5 (add_v5.cu) - Peak Performance with Prefetching**
- **Purpose**: Achieve ultimate performance through advanced memory management
- **Implementation**: Memory prefetching with `cudaMemPrefetchAsync`
- **Key Concepts**: Unified memory optimization, proactive data placement, memory locality
- **Performance**: ~6.5 seconds (**21x total speedup from CPU!**)
- **Learning Focus**: Advanced optimization techniques for production code

## 📊 Performance Results Summary

| Version | Configuration | Runtime | Speedup vs CPU | Key Innovation |
|---------|---------------|---------|-----------------|----------------|
| **v0** | CPU Sequential | ~140s | 1.0x | Baseline Implementation |
| **v1** | 1 GPU Thread | ~120s | 1.2x | CUDA Introduction |
| **v2** | 1×256 Threads | ~12s | **11.7x** | Thread Parallelism |
| **v3** | Many×256 | ~11s | **12.7x** | Full GPU Utilization |
| **v4** | GPU Init | ~7s | **20.0x** | Memory Optimization |
| **v5** | Prefetching | ~6.5s | **21.5x** | Peak Performance |

*Results measured on NVIDIA H100 GPU with 1+ billion elements*

## 🚀 Getting Started

### **Prerequisites**
- Access to h100 partition on SLURM cluster
- CUDA toolkit and compatible GCC compiler
- Basic understanding of C/C++ programming

### **Quick Start - Interactive Testing**
```bash
# Get interactive GPU session
interactive -p h100

# Load modules
module load gcc/14.2.0 cuda

# Run interactive performance analysis
./interactive_test.sh
```

### **Manual Compilation and Testing**
```bash
# Compile all versions
make all

# Run individual versions
./add_v0.exe  # CPU baseline
./add_v1.exe  # Single GPU thread
./add_v2.exe  # Single block parallelism
./add_v3.exe  # Full GPU utilization
./add_v4.exe  # GPU initialization
./add_v5.exe  # Peak performance

# Clean build files
make clean
```

## 🔧 Available Tools and Scripts

### **Analysis Scripts**
- **`interactive_test.sh`** - Interactive compilation, execution, and profiling
- **`submit_job.sh`** - Individual job submission script for specific versions
- **`analyze_performance.sh`** - Local performance testing (requires pre-compiled executables)

### **Build System**
- **`Makefile`** - Compilation rules for all versions
- **`clean.sh`** - Cleanup script for build artifacts
- **`cuda_error_check.h`** - Robust CUDA error checking macros

## 🎓 Educational Learning Outcomes

### **GPU Programming Fundamentals**
✅ CUDA kernel syntax and execution model  
✅ Thread hierarchy (grids, blocks, threads)  
✅ Memory management strategies  
✅ Error checking and debugging practices  

### **Performance Optimization Strategies**
✅ Identifying and eliminating bottlenecks  
✅ Memory access pattern optimization  
✅ Hardware utilization maximization  
✅ Progressive optimization methodology  

### **Advanced CUDA Features**
✅ Unified Memory system and automatic migration  
✅ Memory prefetching techniques  
✅ Multi-kernel execution patterns  
✅ GPU device property utilization  

### **Parallel Programming Concepts**
✅ Sequential vs parallel algorithm design  
✅ Work distribution and load balancing  
✅ Scalability across different hardware  
✅ Performance measurement and analysis  

## 🏗️ Code Architecture

### **Memory Management Evolution**
- **v0**: Standard C++ heap allocation (`new`/`delete`)
- **v1-v5**: CUDA Unified Memory (`cudaMallocManaged`)
- **v5**: Advanced prefetching (`cudaMemPrefetchAsync`)

### **Parallelization Strategy**
- **Thread Indexing**: `threadIdx.x` for intra-block coordination
- **Block Indexing**: `blockIdx.x` for inter-block coordination  
- **Global Indexing**: `blockIdx.x * blockDim.x + threadIdx.x`
- **Grid-Stride Loops**: Scalable processing for any array size

### **Error Handling**
- **CUDA_CHECK Macro**: Comprehensive error checking for all CUDA calls
- **Result Validation**: Mathematical correctness verification
- **Performance Reporting**: Detailed timing and configuration information

## 🔬 Advanced Topics Covered

### **GPU Hardware Understanding**
- Streaming Multiprocessor (SM) architecture
- Thread block scheduling and occupancy
- Memory hierarchy and bandwidth optimization
- Compute capability and hardware limits

### **CUDA Programming Patterns**
- Grid-stride loops for scalability
- Cooperative groups and synchronization
- Memory coalescing patterns
- Kernel launch configuration optimization

### **Performance Optimization Techniques**
- Memory access pattern optimization
- Occupancy maximization strategies
- Memory bandwidth utilization
- Latency hiding through parallelism

## 📝 References and Further Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [GPU Architecture Whitepapers](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c/)

---

**🎉 Congratulations on completing the GPU Programming journey!**
