/*
 * ADD_V5.CU - Memory Prefetching Optimization (Peak Performance)
 * ==============================================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This is the ultimate optimized version that adds memory prefetching
 * to achieve peak performance. It demonstrates advanced unified memory
 * management techniques for optimal GPU-CPU data movement.
 * 
 * OPTIMIZATION STRATEGY:
 * - Memory prefetching using cudaMemPrefetchAsync()
 * - Proactive data placement on GPU before kernel execution
 * - Eliminates on-demand page migration overhead
 * - Optimal memory locality for GPU kernels
 * 
 * CUDA CONFIGURATION:
 * - Same as v4: Multiple blocks with 256 threads each
 * - Two kernel launches: init() then add()
 * - Added: Memory prefetching before kernel execution
 * - Advanced unified memory management
 * 
 * ALGORITHM FLOW:
 * 1. Allocate unified memory
 * 2. Prefetch memory pages to GPU
 * 3. Launch init kernel (data already on GPU)
 * 4. Launch add kernel (optimal memory locality)
 * 5. Validate results on CPU (automatic migration back)
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: CUDA Unified Memory with prefetching
 * - Initialization: GPU-parallel with optimal memory placement
 * - Processing: Full GPU parallelization with prefetched data
 * - Expected runtime: ~3.9-4.0 seconds (excellent performance!)
 * - Memory bandwidth: Optimal utilization of GPU memory system
 * 
 * KEY LEARNING POINTS:
 * 1. Memory prefetching can provide significant performance gains
 * 2. Unified memory requires careful management for optimal performance
 * 3. Proactive data placement vs reactive page migration
 * 4. Advanced CUDA memory management techniques
 * 5. Complete GPU programming optimization workflow
 * 6. Understanding of GPU memory hierarchy and migration costs
 */

#include <iostream>
#include <math.h>
#include "cuda_error_check.h"

/**
 * CUDA Kernel: Optimized GPU-based array initialization
 * ====================================================
 * 
 * Same kernel as v4, but now operates on prefetched memory
 * for optimal performance. Memory is already resident on GPU.
 * 
 * @param n    Number of elements to initialize
 * @param x    Array to initialize with 1.0f values (prefetched)
 * @param y    Array to initialize with 2.0f values (prefetched)
 * 
 * PERFORMANCE ADVANTAGE:
 * - Memory already resident on GPU (no page faults)
 * - No on-demand migration overhead
 * - Optimal memory access patterns
 * - Maximum GPU memory bandwidth utilization
 */
__global__ 
void init(int n, float *x, float *y) {
  // Calculate unique global thread index
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  // Calculate grid stride for grid-stride loop
  int stride = blockDim.x * gridDim.x;
  
  // Initialize elements with prefetched memory
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;  // No page faults - memory already on GPU
    y[i] = 2.0f;  // Optimal memory access performance
  }
}

/**
 * CUDA Kernel: Optimized vector addition computation
 * =================================================
 * 
 * Same kernel as v3 and v4, but now operates on prefetched
 * memory for peak performance.
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand, prefetched)
 * @param y    Input/output array (second operand, stores result, prefetched)
 */
__global__
void add(int n, float *x, float *y) {
  // Calculate unique global thread index
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Calculate grid stride for grid-stride loop
  int stride = blockDim.x * gridDim.x;
  
  // Process elements with optimal memory locality
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];  // Peak performance with prefetched data
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
  
  std::cout << "CUDA v5: Memory Prefetching Optimization (Peak Performance)" << std::endl;
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
  // ADVANCED MEMORY MANAGEMENT: PREFETCHING
  // ============================================================================
  std::cout << std::endl << "=== MEMORY PREFETCHING OPTIMIZATION ===" << std::endl;
  
  // Get current GPU device ID for prefetching
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  std::cout << "Current GPU device: " << device << std::endl;
  
  // Get GPU device properties for memory information
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Global memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
  
  // Prefetch memory pages to GPU BEFORE any kernel execution
  // This moves memory pages from CPU to GPU proactively
  std::cout << "Prefetching memory to GPU..." << std::endl;
  
  CUDA_CHECK(cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL));
  CUDA_CHECK(cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL));
  
  // Wait for prefetching to complete
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Memory prefetching complete - data now resident on GPU." << std::endl;
  
  // TECHNICAL EXPLANATION:
  // cudaMemPrefetchAsync() moves unified memory pages to the specified device
  // This eliminates page faults and on-demand migration during kernel execution
  // Result: Better memory bandwidth utilization and lower latency
  
  // ============================================================================
  // PHASE 1: OPTIMIZED GPU INITIALIZATION
  // ============================================================================
  std::cout << std::endl << "Phase 1: GPU initialization with prefetched memory..." << std::endl;
  
  // Launch initialization kernel (memory already on GPU)
  init<<<numBlocks, blockSize>>>(N, x, y);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  
  // Wait for initialization to complete
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "GPU initialization complete (optimal memory performance)." << std::endl;
  
  // ============================================================================
  // PHASE 2: OPTIMIZED COMPUTATION
  // ============================================================================
  std::cout << std::endl << "Phase 2: GPU computation with prefetched memory..." << std::endl;
  
  // Launch computation kernel (memory already optimally placed)
  add<<<numBlocks, blockSize>>>(N, x, y);
  
  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  // ============================================================================
  // GPU SYNCHRONIZATION & COMPLETION
  // ============================================================================
  std::cout << "Waiting for GPU computation to complete..." << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());

  // ============================================================================
  // RESULT VALIDATION (AUTOMATIC MIGRATION TO CPU)
  // ============================================================================
  std::cout << std::endl << "Phase 3: Result validation..." << std::endl;
  std::cout << "Note: Unified memory will automatically migrate data back to CPU for validation." << std::endl;
  
  float maxError = 0.0f;        // Track maximum error found
  float tolerance = 1e-5f;      // Acceptable floating-point error
  int errorCount = 0;           // Count of elements outside tolerance
  
  // CPU accesses will trigger automatic migration from GPU to CPU
  // This is handled transparently by the CUDA unified memory system
  for (int i = 0; i < N; i++) {
    float error = fabs(y[i] - 3.0f);  // Calculate absolute error
    maxError = fmax(maxError, error); // Track maximum error
    if (error > tolerance) errorCount++;  // Count significant errors
  }
  
  // ============================================================================
  // FINAL RESULTS & PERFORMANCE SUMMARY
  // ============================================================================
  std::cout << std::endl << "=== CUDA v5 PEAK PERFORMANCE RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
  std::cout << "GPU configuration: " << numBlocks << " blocks × " << blockSize 
            << " threads = " << totalThreads << " total threads" << std::endl;
  std::cout << "Memory optimization: Prefetching enabled" << std::endl;
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
  
  // ============================================================================
  // COMPREHENSIVE PERFORMANCE ANALYSIS
  // ============================================================================
  std::cout << std::endl << "=== PEAK PERFORMANCE ACHIEVED ===" << std::endl;
  std::cout << "This version achieves optimal performance through:" << std::endl;
  std::cout << "✓ Full GPU parallelization (multiple blocks)" << std::endl;
  std::cout << "✓ GPU-based initialization (eliminates CPU bottleneck)" << std::endl;
  std::cout << "✓ Memory prefetching (optimal data locality)" << std::endl;
  std::cout << "✓ Grid-stride loops (scalable to any data size)" << std::endl;
  std::cout << "✓ Unified memory management (simplified programming)" << std::endl;
  
  std::cout << std::endl << "=== COMPLETE PERFORMANCE PROGRESSION ===" << std::endl;
  std::cout << "v0 (CPU):        ~140s  - Sequential CPU baseline" << std::endl;
  std::cout << "v1 (1 thread):   ~120s  - GPU overhead without parallelism" << std::endl;
  std::cout << "v2 (1 block):     ~12s  - Thread-level parallelism (10x speedup)" << std::endl;
  std::cout << "v3 (many blocks): ~11s  - Full GPU utilization" << std::endl;
  std::cout << "v4 (GPU init):     ~7s  - Memory initialization optimization" << std::endl;
  std::cout << "v5 (prefetch):    ~3.9s - Peak performance with prefetching (35x speedup!)" << std::endl;
  
  std::cout << std::endl << "KEY LEARNING OUTCOMES:" << std::endl;
  std::cout << "• GPU programming requires proper parallelization strategy" << std::endl;
  std::cout << "• Memory management is crucial for GPU performance" << std::endl;
  std::cout << "• Progressive optimization leads to dramatic improvements" << std::endl;
  std::cout << "• Understanding hardware architecture drives optimization" << std::endl;
  std::cout << "• Profiling and measurement guide optimization decisions" << std::endl;

  // ============================================================================
  // MEMORY CLEANUP
  // ============================================================================
  std::cout << std::endl << "Cleaning up unified memory..." << std::endl;
  CUDA_CHECK(cudaFree(x)); 
  CUDA_CHECK(cudaFree(y));
  
  std::cout << std::endl << "🚀 CUDA v5 PEAK PERFORMANCE COMPUTATION COMPLETE! 🚀" << std::endl;
  
  return 0;
}
