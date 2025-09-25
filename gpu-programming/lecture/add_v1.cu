/*
 * ADD_V1.CU - Single GPU Thread Implementation
 * ===========================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This is the first CUDA implementation using only ONE thread on the GPU.
 * It demonstrates why simply moving code to GPU doesn't guarantee performance gains.
 * This version is intentionally inefficient to illustrate GPU programming concepts.
 * 
 * CUDA CONFIGURATION:
 * - Grid size: 1 block
 * - Block size: 1 thread
 * - Total threads: 1 × 1 = 1 thread
 * - Thread indexing: Not needed (only one thread)
 * 
 * ALGORITHM:
 * - Sequential processing on GPU (similar to CPU version)
 * - Single thread processes ALL elements sequentially
 * - Uses CUDA Unified Memory for simplified memory management
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: CUDA Unified Memory (cudaMallocManaged)
 * - Processing: Sequential loop on GPU (single thread)
 * - Expected runtime: ~120 seconds (slower than CPU due to GPU overhead!)
 * - GPU utilization: Extremely poor (~0.1%)
 * 
 * KEY LEARNING POINTS:
 * 1. GPU isn't automatically faster - requires proper parallelization
 * 2. Single thread on GPU has overhead compared to CPU
 * 3. Unified Memory simplifies memory management but has performance implications
 * 4. Proper synchronization is essential for correctness
 * 5. Introduction to CUDA kernel launch syntax: <<<blocks, threads>>>
 */

#include <iostream>
#include <math.h>
#include "cuda_error_check.h"

/**
 * CUDA Kernel: Single-threaded vector addition
 * ============================================
 * 
 * __global__ qualifier indicates this function runs on GPU
 * and can be called from CPU code (host code).
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand)
 * @param y    Input/output array (second operand, stores result)
 * 
 * PERFORMANCE ISSUE:
 * This kernel uses only one thread to process ALL elements sequentially.
 * This is extremely inefficient for GPU architecture, which is designed
 * for parallel processing with thousands of threads.
 * 
 * WHY THIS IS SLOW:
 * - Only 1 out of thousands of GPU cores is utilized
 * - Sequential processing doesn't leverage GPU's parallel architecture
 * - GPU thread scheduling overhead without parallelism benefits
 * - Memory access patterns are not optimized for GPU
 */
__global__
void add(int n, float *x, float *y) {
  // This loop runs entirely on a single GPU thread
  // All GPU's parallel processing power is wasted!
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];  // Sequential element-wise addition
  }
  
  // Note: No thread indexing needed since only one thread exists
  // In parallel versions, we'll use threadIdx.x and blockIdx.x
}

int main(void) {
  // ============================================================================
  // ARRAY SIZE CONFIGURATION
  // ============================================================================
  int N = 1<<30; // 2^30 = 1,073,741,824 elements (~8.6GB total memory)
  // int N = 1<<28; // 2^28 = 268,435,456 elements (~2.1GB - alternative size)
  // int N = 1<<25; // 2^25 = 33,554,432 elements (~268MB - original size)  
  // int N = 1<<20; // 2^20 = 1,048,576 elements (~8.4MB - small test size)
  
  std::cout << "CUDA v1: Single Thread Implementation" << std::endl;
  std::cout << "Processing " << N << " elements (" << (N * sizeof(float) * 2) / (1024*1024*1024) << " GB total)" << std::endl;
  
  float *x, *y;  // Pointers will point to unified memory

  // ============================================================================
  // MEMORY ALLOCATION (UNIFIED MEMORY)
  // ============================================================================
  // CUDA Unified Memory allows same pointers to be accessed from CPU and GPU
  // The CUDA runtime handles data movement automatically between CPU and GPU
  // This simplifies programming but may have performance implications
  std::cout << "Allocating Unified Memory..." << std::endl;
  
  CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(float)));  // Allocate array x
  CUDA_CHECK(cudaMallocManaged(&y, N*sizeof(float)));  // Allocate array y
  
  // Memory is now accessible from both CPU (host) and GPU (device)

  // ============================================================================
  // DATA INITIALIZATION (ON CPU)
  // ============================================================================
  // Initialize arrays on CPU side of unified memory
  std::cout << "Initializing arrays on CPU..." << std::endl;
  
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;  // Set all elements of x to 1.0
    y[i] = 2.0f;  // Set all elements of y to 2.0
  }

  // ============================================================================
  // GPU SYNCHRONIZATION
  // ============================================================================
  // Ensure CPU initialization completes before GPU kernel launch
  // This is crucial for correctness with unified memory
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // KERNEL LAUNCH CONFIGURATION
  // ============================================================================
  std::cout << "Launching CUDA kernel with 1 block, 1 thread..." << std::endl;
  
  // CUDA kernel launch syntax: kernel_name<<<grid_size, block_size>>>(parameters)
  // <<<1, 1>>> means:
  //   - 1 block in the grid
  //   - 1 thread per block  
  //   - Total threads = 1 × 1 = 1 thread
  add<<<1, 1>>>(N, x, y);
  
  // Check for kernel launch errors (configuration errors, etc.)
  CUDA_CHECK(cudaGetLastError());

  // ============================================================================
  // GPU SYNCHRONIZATION & COMPLETION
  // ============================================================================
  // Wait for GPU kernel to complete before CPU accesses results
  // This is essential for correctness in GPU programming
  std::cout << "Waiting for GPU computation to complete..." << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // RESULT VALIDATION
  // ============================================================================
  // Verify computational correctness by checking results on CPU
  std::cout << "Validating results..." << std::endl;
  
  float maxError = 0.0f;        // Track maximum error found
  float tolerance = 1e-5f;      // Acceptable floating-point error  
  int errorCount = 0;           // Count of elements outside tolerance
  
  // Check every element for correctness
  for (int i = 0; i < N; i++) {
    float error = fabs(y[i] - 3.0f);      // Calculate absolute error
    maxError = fmax(maxError, error);     // Track maximum error
    if (error > tolerance) errorCount++;  // Count significant errors
  }
  
  // ============================================================================
  // RESULTS REPORTING
  // ============================================================================
  std::cout << std::endl << "=== CUDA v1 SINGLE THREAD RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
  std::cout << "GPU configuration: 1 block × 1 thread = 1 total thread" << std::endl;
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
  std::cout << std::endl << "PERFORMANCE INSIGHT:" << std::endl;
  std::cout << "This version is slower than CPU because:" << std::endl;
  std::cout << "- Only 1 GPU thread used (massive underutilization)" << std::endl;
  std::cout << "- GPU kernel launch overhead without parallel benefit" << std::endl;
  std::cout << "- Sequential processing doesn't leverage GPU architecture" << std::endl;
  std::cout << "Next version (v2) will introduce thread-level parallelism!" << std::endl;

  // ============================================================================
  // MEMORY CLEANUP
  // ============================================================================
  // Free unified memory allocated with cudaMallocManaged
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  
  std::cout << "CUDA v1 single thread computation complete." << std::endl;
  return 0;
}
