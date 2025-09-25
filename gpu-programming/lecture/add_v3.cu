/*
 * ADD_V3.CU - Multiple Blocks (Full GPU Parallelization)
 * ======================================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture  
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This version achieves full GPU utilization by using multiple blocks
 * across all available Streaming Multiprocessors (SMs). It represents
 * the first "properly parallel" CUDA implementation that leverages
 * the GPU's complete parallel architecture.
 * 
 * CUDA CONFIGURATION:
 * - Grid size: Calculated to cover all data (typically 4M+ blocks)
 * - Block size: 256 threads per block  
 * - Total threads: numBlocks × 256 (often > 1 billion threads!)
 * - Thread indexing: Global index using blockIdx.x and threadIdx.x
 * 
 * ALGORITHM:
 * - Each thread gets a unique global index across all blocks
 * - Threads process elements with a grid-stride loop
 * - Grid-stride allows handling arrays larger than thread count
 * - All GPU Streaming Multiprocessors are utilized
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: CUDA Unified Memory
 * - Processing: Full GPU parallelization across all SMs
 * - Expected runtime: ~11 seconds (similar to v2 but more scalable)
 * - GPU utilization: High (80-90%+ of available SMs)
 * 
 * KEY LEARNING POINTS:
 * 1. Block-level parallelism enables full GPU utilization
 * 2. Global thread indexing across multiple blocks  
 * 3. Grid-stride loops for processing large datasets
 * 4. Proper block size calculation for optimal performance
 * 5. Understanding the relationship between blocks and SMs
 */

#include <iostream>
#include <math.h>
#include "cuda_error_check.h"

/**
 * CUDA Kernel: Multiple blocks, full GPU utilization
 * ==================================================
 * 
 * This kernel uses multiple blocks to fully utilize all available
 * Streaming Multiprocessors on the GPU. Each thread gets a unique
 * global index and processes elements using a grid-stride pattern.
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand)  
 * @param y    Input/output array (second operand, stores result)
 * 
 * THREAD INDEXING:
 * - threadIdx.x: Thread index within current block (0 to blockDim.x-1)
 * - blockIdx.x:  Block index within grid (0 to gridDim.x-1)
 * - blockDim.x:  Number of threads per block (256)
 * - gridDim.x:   Number of blocks in grid (calculated in main)
 * 
 * GLOBAL THREAD ID CALCULATION:
 * index = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Example with 3 blocks of 4 threads each:
 * Block 0: threads 0, 1, 2, 3    (blockIdx.x=0, threadIdx.x=0,1,2,3)
 * Block 1: threads 4, 5, 6, 7    (blockIdx.x=1, threadIdx.x=0,1,2,3)  
 * Block 2: threads 8, 9, 10, 11  (blockIdx.x=2, threadIdx.x=0,1,2,3)
 * 
 * GRID-STRIDE LOOP:
 * Allows processing arrays larger than the total number of threads
 * by having each thread process multiple elements spaced by grid size.
 */
__global__
void add(int n, float *x, float *y) {
  // Calculate unique global thread index
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Calculate grid stride (total number of threads in the grid)
  int stride = blockDim.x * gridDim.x;
  
  // Grid-stride loop: each thread processes multiple elements
  // Thread 0 processes elements: 0, stride, 2*stride, 3*stride, ...
  // Thread 1 processes elements: 1, 1+stride, 1+2*stride, ...
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];  // Parallel element-wise addition
  }
  
  // Note: Grid-stride pattern ensures:
  // 1. All array elements are processed exactly once
  // 2. Work is evenly distributed across threads
  // 3. Algorithm works for any array size (even > thread count)
}

int main(void) {
  // ============================================================================
  // ARRAY SIZE CONFIGURATION
  // ============================================================================
  int N = 1<<30; // 2^30 = 1,073,741,824 elements (~8.6GB total memory)
  // int N = 1<<28; // 2^28 = 268,435,456 elements (~2.1GB - alternative size)
  // int N = 1<<25; // 2^25 = 33,554,432 elements (~268MB - original size)
  // int N = 1<<20; // 2^20 = 1,048,576 elements (~8.4MB - small test size)
  
  std::cout << "CUDA v3: Multiple Blocks (Full GPU)" << std::endl;
  std::cout << "Processing " << N << " elements (" << (N * sizeof(float) * 2) / (1024*1024*1024) << " GB total)" << std::endl;
  
  float *x, *y;  // Unified memory pointers

  // ============================================================================
  // MEMORY ALLOCATION (UNIFIED MEMORY)
  // ============================================================================
  std::cout << "Allocating Unified Memory..." << std::endl;
  CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N*sizeof(float)));

  // ============================================================================
  // DATA INITIALIZATION (ON CPU)
  // ============================================================================
  std::cout << "Initializing arrays on CPU..." << std::endl;
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;  // Set all elements of x to 1.0
    y[i] = 2.0f;  // Set all elements of y to 2.0
  }

  // ============================================================================
  // KERNEL LAUNCH CONFIGURATION CALCULATION
  // ============================================================================
  // Calculate optimal grid configuration for full GPU utilization
  int blockSize = 256;  // Threads per block (common choice: 128, 256, 512)
  
  // Calculate number of blocks needed to cover all elements
  // Using ceiling division: (N + blockSize - 1) / blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;
  
  // Calculate total threads that will be launched
  long long totalThreads = (long long)numBlocks * blockSize;
  
  std::cout << std::endl << "=== KERNEL LAUNCH CONFIGURATION ===" << std::endl;
  std::cout << "Block size: " << blockSize << " threads per block" << std::endl;
  std::cout << "Grid size: " << numBlocks << " blocks" << std::endl;
  std::cout << "Total threads: " << totalThreads << " threads" << std::endl;
  std::cout << "Elements per thread (avg): " << (double)N / totalThreads << std::endl;
  
  // GPU hardware utilization estimate
  int deviceId = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
  int activeBlocks = min(numBlocks, prop.multiProcessorCount * maxBlocksPerSM);
  
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Estimated active blocks: " << activeBlocks << std::endl;
  std::cout << "Theoretical SM utilization: " << 
                (100.0 * activeBlocks) / (prop.multiProcessorCount * maxBlocksPerSM) << "%" << std::endl;

  // ============================================================================
  // GPU SYNCHRONIZATION 
  // ============================================================================
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // KERNEL LAUNCH
  // ============================================================================
  std::cout << std::endl << "Launching CUDA kernel with full GPU utilization..." << std::endl;
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
  std::cout << "Validating results..." << std::endl;
  
  float maxError = 0.0f;        // Track maximum error found  
  float tolerance = 1e-5f;      // Acceptable floating-point error
  int errorCount = 0;           // Count of elements outside tolerance
  
  // Verify all elements computed correctly
  for (int i = 0; i < N; i++) {
    float error = fabs(y[i] - 3.0f);      // Calculate absolute error
    maxError = fmax(maxError, error);     // Track maximum error
    if (error > tolerance) errorCount++;  // Count significant errors
  }
  
  // ============================================================================
  // RESULTS REPORTING
  // ============================================================================
  std::cout << std::endl << "=== CUDA v3 MULTIPLE BLOCKS RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
  std::cout << "GPU configuration: " << numBlocks << " blocks × " << blockSize 
            << " threads = " << totalThreads << " total threads" << std::endl;
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
  std::cout << std::endl << "PERFORMANCE ACHIEVEMENT:" << std::endl;
  std::cout << "This version achieves full GPU utilization because:" << std::endl;
  std::cout << "- Multiple blocks utilize all " << prop.multiProcessorCount << " Streaming Multiprocessors" << std::endl;
  std::cout << "- Grid-stride loop handles any array size efficiently" << std::endl;
  std::cout << "- Global thread indexing ensures proper work distribution" << std::endl;
  std::cout << "- Scales with GPU hardware capabilities" << std::endl;
  std::cout << std::endl << "NEXT OPTIMIZATIONS:" << std::endl;
  std::cout << "- v4: GPU-based initialization (reduce CPU-GPU data transfer)" << std::endl;
  std::cout << "- v5: Memory prefetching (optimize unified memory performance)" << std::endl;

  // ============================================================================
  // MEMORY CLEANUP
  // ============================================================================
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  
  std::cout << "CUDA v3 multiple blocks computation complete." << std::endl;
  return 0;
}
