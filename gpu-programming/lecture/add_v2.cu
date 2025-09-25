/*
 * ADD_V2.CU - Single Block with Multiple Threads
 * ==============================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This version introduces thread-level parallelism within a single CUDA block.
 * It demonstrates how multiple threads can work together to process data
 * and shows the first significant GPU performance improvement.
 * 
 * CUDA CONFIGURATION:
 * - Grid size: 1 block
 * - Block size: 256 threads
 * - Total threads: 1 × 256 = 256 threads
 * - Thread indexing: threadIdx.x (0 to 255)
 * 
 * ALGORITHM:
 * - Each thread processes every 256th element (strided access)
 * - Thread 0 processes elements 0, 256, 512, 768, ...
 * - Thread 1 processes elements 1, 257, 513, 769, ...
 * - Thread i processes elements i, i+256, i+512, i+768, ...
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: CUDA Unified Memory
 * - Processing: 256-way parallel processing on GPU
 * - Expected runtime: ~12 seconds (10x+ speedup vs v1!)
 * - GPU utilization: Low (~1%) - only one SM utilized
 * 
 * KEY LEARNING POINTS:
 * 1. Thread-level parallelism provides significant speedup
 * 2. Strided memory access pattern for work distribution
 * 3. Thread indexing with threadIdx.x
 * 4. Single block limits utilization to one Streaming Multiprocessor
 * 5. Introduction to the concept of "stride" in parallel algorithms
 */

#include <iostream>
#include <math.h>
#include "cuda_error_check.h"

/**
 * CUDA Kernel: Single block, multiple threads vector addition
 * ==========================================================
 * 
 * This kernel uses multiple threads within a single block to process
 * the array in parallel. Each thread handles every Nth element where
 * N is the number of threads in the block (blockDim.x).
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand)
 * @param y    Input/output array (second operand, stores result)
 * 
 * THREAD ORGANIZATION:
 * - threadIdx.x: Thread index within block (0 to blockDim.x-1)
 * - blockDim.x:  Number of threads in block (256 in this case)
 * - stride:      Step size between elements each thread processes
 * 
 * WORK DISTRIBUTION PATTERN:
 * Thread 0:   processes elements 0,   256, 512, 768, ...
 * Thread 1:   processes elements 1,   257, 513, 769, ...
 * Thread 2:   processes elements 2,   258, 514, 770, ...
 * ...
 * Thread 255: processes elements 255, 511, 767, 1023, ...
 */
__global__
void add(int n, float *x, float *y){
  // Calculate this thread's starting index and stride
  int index = threadIdx.x;    // Thread ID within block (0-255)
  int stride = blockDim.x;    // Number of threads in block (256)
  
  // Each thread processes every 'stride'th element
  // This creates a strided access pattern across the array
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];  // Parallel element-wise addition
  }
  
  // Note: No synchronization needed within loop since each thread
  // works on different array elements (no data dependencies)
}

int main(void) {
  // ============================================================================
  // ARRAY SIZE CONFIGURATION  
  // ============================================================================
  int N = 1<<30; // 2^30 = 1,073,741,824 elements (~8.6GB total memory)
  // int N = 1<<28; // 2^28 = 268,435,456 elements (~2.1GB - alternative size)
  // int N = 1<<25; // 2^25 = 33,554,432 elements (~268MB - original size)
  // int N = 1<<20; // 2^20 = 1,048,576 elements (~8.4MB - small test size)
  
  std::cout << "CUDA v2: Single Block (256 threads)" << std::endl;
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
  // GPU SYNCHRONIZATION
  // ============================================================================
  // Ensure CPU initialization completes before GPU kernel launch
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // KERNEL LAUNCH CONFIGURATION
  // ============================================================================
  const int BLOCK_SIZE = 256;  // Threads per block
  const int GRID_SIZE = 1;     // Number of blocks
  
  std::cout << "Launching CUDA kernel:" << std::endl;
  std::cout << "  Grid size: " << GRID_SIZE << " blocks" << std::endl;
  std::cout << "  Block size: " << BLOCK_SIZE << " threads per block" << std::endl;
  std::cout << "  Total threads: " << (GRID_SIZE * BLOCK_SIZE) << std::endl;
  std::cout << "  Work per thread: ~" << (N / (GRID_SIZE * BLOCK_SIZE)) << " elements" << std::endl;
  
  // Launch kernel with 1 block of 256 threads
  // <<<1, 256>>> means:
  //   - 1 block in the grid
  //   - 256 threads per block
  //   - Total threads = 1 × 256 = 256 threads
  add<<<GRID_SIZE, BLOCK_SIZE>>>(N, x, y);
  
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
  std::cout << std::endl << "=== CUDA v2 SINGLE BLOCK RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
  std::cout << "GPU configuration: " << GRID_SIZE << " block × " << BLOCK_SIZE << " threads = " 
            << (GRID_SIZE * BLOCK_SIZE) << " total threads" << std::endl;
  std::cout << "Max error: " << maxError << std::endl;
  std::cout << "Validation: " << (maxError <= tolerance ? "PASS" : "FAIL") << std::endl;
  
  if (errorCount > 0) {
    std::cout << "Elements outside tolerance (" << tolerance << "): " 
              << errorCount << " / " << N 
              << " (" << (100.0 * errorCount / N) << "%)" << std::endl;
  } else {
    std::cout << "All elements computed correctly!" << std::endl;
  }
  
  // Performance insight
  std::cout << std::endl << "PERFORMANCE IMPROVEMENT:" << std::endl;
  std::cout << "This version shows significant speedup because:" << std::endl;
  std::cout << "- 256 threads work in parallel (vs 1 thread in v1)" << std::endl;
  std::cout << "- Each thread processes ~" << (N/256) << " elements" << std::endl;
  std::cout << "- Thread-level parallelism utilized within single block" << std::endl;
  std::cout << std::endl << "REMAINING LIMITATION:" << std::endl;
  std::cout << "- Only 1 block used (1 Streaming Multiprocessor out of many)" << std::endl;
  std::cout << "- Next version (v3) will use multiple blocks for full GPU utilization!" << std::endl;

  // ============================================================================
  // MEMORY CLEANUP
  // ============================================================================
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  
  std::cout << "CUDA v2 single block computation complete." << std::endl;
  return 0;
}
