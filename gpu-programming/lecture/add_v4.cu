/*
 * ADD_V4.CU - GPU-Based Initialization Optimization
 * =================================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This version optimizes memory initialization by moving it to the GPU.
 * Instead of initializing arrays on CPU and transferring to GPU, we
 * initialize directly on GPU, reducing memory transfer overhead.
 * 
 * OPTIMIZATION STRATEGY:
 * - Separate GPU kernel for array initialization
 * - Both initialization and computation happen on GPU
 * - Reduces CPU-GPU memory transfer time
 * - Better utilization of GPU memory bandwidth
 * 
 * CUDA CONFIGURATION:
 * - Grid size: Calculated to cover all data elements
 * - Block size: 256 threads per block
 * - Two kernel launches: init() then add()
 * - Same parallelization strategy as v3
 * 
 * ALGORITHM FLOW:
 * 1. Allocate unified memory
 * 2. Launch init kernel to initialize arrays on GPU
 * 3. Launch add kernel to perform computation on GPU
 * 4. Validate results on CPU
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: CUDA Unified Memory
 * - Initialization: GPU-parallel (vs CPU-sequential in v1-v3)
 * - Processing: Full GPU parallelization
 * - Expected runtime: ~7 seconds (35% improvement from v3!)
 * - Memory bandwidth: Better GPU memory utilization
 * 
 * KEY LEARNING POINTS:
 * 1. GPU kernels can be used for more than just computation
 * 2. Memory initialization is often a significant bottleneck
 * 3. Multiple kernel launches in sequence
 * 4. GPU memory bandwidth optimization techniques
 * 5. Unified memory performance considerations
 */

#include <iostream>
#include <math.h>
#include "cuda_error_check.h"

/**
 * CUDA Kernel: GPU-based array initialization
 * ===========================================
 * 
 * This kernel initializes both input arrays directly on the GPU,
 * eliminating the need for CPU initialization and data transfer.
 * Uses the same parallel pattern as the computation kernel.
 * 
 * @param n    Number of elements to initialize
 * @param x    Array to initialize with 1.0f values
 * @param y    Array to initialize with 2.0f values
 * 
 * PERFORMANCE BENEFIT:
 * - GPU parallel initialization vs CPU sequential initialization
 * - Eliminates CPU-to-GPU memory transfer for initial data
 * - Better memory bandwidth utilization
 * - Reduces overall execution time significantly
 * 
 * THREAD ORGANIZATION:
 * Same grid-stride pattern as computation kernel ensures
 * optimal memory access patterns and thread utilization.
 */
__global__ 
void init(int n, float *x, float *y) {
    // Calculate unique global thread index
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate grid stride for grid-stride loop
    int stride = blockDim.x * gridDim.x;
    
    // Each thread initializes multiple elements in parallel
    for (int i = index; i < n; i += stride) {
        x[i] = 1.0f;  // Initialize first array
        y[i] = 2.0f;  // Initialize second array
    }
    
    // Note: This parallel initialization is much faster than
    // sequential CPU initialization for large arrays
}

/**
 * CUDA Kernel: Vector addition computation
 * =======================================
 * 
 * Same kernel as v3 - multiple blocks with grid-stride loop
 * for optimal GPU utilization and scalability.
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand)
 * @param y    Input/output array (second operand, stores result)
 */
__global__
void add(int n, float *x, float *y) {
  // Calculate unique global thread index
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Calculate grid stride for grid-stride loop
  int stride = blockDim.x * gridDim.x;
  
  // Grid-stride loop for processing elements
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];  // Element-wise addition
  }
}

int main(void) {
  // ============================================================================
  // ARRAY SIZE CONFIGURATION
  // ============================================================================
  int N = 1<<30; // 2^30 = 1,073,741,824 elements (~8.6GB total memory)
  // int N = 1<<28; // 2^28 = 268,435,456 elements (~2.1GB - alternative size)
  // int N = 1<<25; // 2^25 = 33,554,432 elements (~268MB - original size)
  // int N = 1<<20; // 2^20 = 1,048,576 elements (~8.4MB - small test size)
  
  std::cout << "CUDA v4: GPU Initialization Optimization" << std::endl;
  std::cout << "Processing " << N << " elements (" << (N * sizeof(float) * 2) / (1024*1024*1024) << " GB total)" << std::endl;
  
  float *x, *y;  // Unified memory pointers

  // ============================================================================
  // MEMORY ALLOCATION (UNIFIED MEMORY)
  // ============================================================================
  std::cout << "Allocating Unified Memory..." << std::endl;
  CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N*sizeof(float)));

  // ============================================================================
  // KERNEL LAUNCH CONFIGURATION
  // ============================================================================
  int blockSize = 256;  // Threads per block
  int numBlocks = (N + blockSize - 1) / blockSize;  // Blocks needed
  
  long long totalThreads = (long long)numBlocks * blockSize;
  
  std::cout << std::endl << "=== KERNEL CONFIGURATION ===" << std::endl;
  std::cout << "Block size: " << blockSize << " threads per block" << std::endl;
  std::cout << "Grid size: " << numBlocks << " blocks" << std::endl;
  std::cout << "Total threads: " << totalThreads << " threads" << std::endl;
  
  // ============================================================================
  // PHASE 1: GPU-BASED INITIALIZATION
  // ============================================================================
  std::cout << std::endl << "Phase 1: GPU-based array initialization..." << std::endl;
  
  // Launch initialization kernel on GPU instead of CPU loop
  init<<<numBlocks, blockSize>>>(N, x, y);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  
  // Wait for initialization to complete before launching computation
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "GPU initialization complete." << std::endl;
  
  // ============================================================================
  // PHASE 2: COMPUTATION
  // ============================================================================
  std::cout << std::endl << "Phase 2: GPU computation..." << std::endl;
  
  // Launch computation kernel
  add<<<numBlocks, blockSize>>>(N, x, y);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  // ============================================================================
  // GPU SYNCHRONIZATION & COMPLETION
  // ============================================================================
  std::cout << "Waiting for GPU computation to complete..." << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // RESULT VALIDATION
  // ============================================================================
  std::cout << std::endl << "Phase 3: Result validation..." << std::endl;
  
  float maxError = 0.0f;        // Track maximum error found
  float tolerance = 1e-5f;      // Acceptable floating-point error
  int errorCount = 0;           // Count of elements outside tolerance
  
  // Verify all elements computed correctly
  for (int i = 0; i < N; i++) {
    float error = fabs(y[i] - 3.0f);  // Calculate absolute error
    maxError = fmax(maxError, error); // Track maximum error
    if (error > tolerance) errorCount++;  // Count significant errors
  }
  
  // ============================================================================
  // RESULTS REPORTING  
  // ============================================================================
  std::cout << std::endl << "=== CUDA v4 GPU INITIALIZATION RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
  std::cout << "GPU configuration: " << numBlocks << " blocks × " << blockSize 
            << " threads = " << totalThreads << " total threads" << std::endl;
  std::cout << "Kernels launched: 2 (init + add)" << std::endl;
  std::cout << "Max error: " << maxError << std::endl;
  std::cout << "Validation: " << (maxError <= tolerance ? "PASS" : "FAIL") << std::endl;
  
  if (errorCount > 0) {
    std::cout << "Elements outside tolerance (" << tolerance << "): " 
              << errorCount << " / " << N 
              << " (" << (100.0 * errorCount / N) << "%)" << std::endl;
  } else {
    std::cout << "All elements computed correctly!" << std::endl;
  }
  
  // Performance insights
  std::cout << std::endl << "OPTIMIZATION IMPACT:" << std::endl;
  std::cout << "GPU initialization provides significant speedup because:" << std::endl;
  std::cout << "- Parallel initialization vs sequential CPU initialization" << std::endl;
  std::cout << "- Eliminates CPU-to-GPU memory transfer overhead" << std::endl;
  std::cout << "- Better utilization of GPU memory bandwidth" << std::endl;
  std::cout << "- Two-phase GPU execution (init + compute)" << std::endl;
  std::cout << std::endl << "PERFORMANCE PROGRESSION:" << std::endl;
  std::cout << "- v1: ~120s (single GPU thread)" << std::endl;
  std::cout << "- v2: ~12s (single block parallelism)" << std::endl;
  std::cout << "- v3: ~11s (full GPU parallelism)" << std::endl;
  std::cout << "- v4: ~7s (GPU initialization optimization)" << std::endl;
  std::cout << "- Next: v5 will add memory prefetching for further optimization" << std::endl;

  // ============================================================================
  // MEMORY CLEANUP
  // ============================================================================
  CUDA_CHECK(cudaFree(x)); 
  CUDA_CHECK(cudaFree(y));
  
  std::cout << "CUDA v4 GPU initialization computation complete." << std::endl;
  return 0;
}
