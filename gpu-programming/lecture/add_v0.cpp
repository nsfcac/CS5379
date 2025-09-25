/*
 * ADD_V0.CPP - CPU Baseline Implementation
 * ========================================
 * 
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * PURPOSE:
 * This is the baseline CPU implementation for vector addition.
 * It serves as a reference point to compare GPU performance against.
 * 
 * ALGORITHM:
 * - Sequential vector addition: y[i] = x[i] + y[i]
 * - Pure CPU implementation using standard C++
 * - Single-threaded execution (no parallelism)
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Memory allocation: Standard heap allocation (new/delete)
 * - Processing: Sequential loop on CPU
 * - Expected runtime: ~140 seconds for 1B+ elements
 * - Memory usage: ~8.6GB for arrays
 * 
 * KEY LEARNING POINTS:
 * 1. Establishes baseline performance for comparison
 * 2. Shows traditional CPU approach to array processing
 * 3. Demonstrates validation techniques for correctness checking
 * 4. Highlights memory management in standard C++
 */

#include <iostream>
#include <math.h>

/**
 * CPU function to add corresponding elements of two arrays
 * 
 * @param n    Number of elements to process
 * @param x    Input array (first operand)
 * @param y    Input/output array (second operand, stores result)
 * 
 * This function performs element-wise addition: y[i] = x[i] + y[i]
 * It runs sequentially on the CPU using a simple for loop.
 */
void add(int n, float *x, float *y) {
  // Sequential processing - each element computed one at a time
  // This is the traditional CPU approach with no parallelism
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];  // Element-wise addition
  }
}

int main(void) {
  // ============================================================================
  // ARRAY SIZE CONFIGURATION
  // ============================================================================
  // Using bit shifts for power-of-2 sizes (common in GPU programming)
  int N = 1<<30; // 2^30 = 1,073,741,824 elements (~8.6GB total memory)
  // int N = 1<<28; // 2^28 = 268,435,456 elements (~2.1GB - alternative size)
  // int N = 1<<25; // 2^25 = 33,554,432 elements (~268MB - original size)
  // int N = 1<<20; // 2^20 = 1,048,576 elements (~8.4MB - small test size)
  
  std::cout << "Processing " << N << " elements (" << (N * sizeof(float) * 2) / (1024*1024*1024) << " GB total)" << std::endl;

  // ============================================================================
  // MEMORY ALLOCATION (CPU)
  // ============================================================================
  // Allocate arrays on CPU heap using standard C++ new operator
  // This memory resides in system RAM, accessible only by CPU
  float *x = new float[N];  // First input array
  float *y = new float[N];  // Second input array (also stores result)

  // ============================================================================
  // DATA INITIALIZATION
  // ============================================================================
  // Initialize arrays with known values for easy validation
  // x[i] = 1.0, y[i] = 2.0, so expected result is y[i] = 3.0
  std::cout << "Initializing arrays on CPU..." << std::endl;
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;  // Set all elements of x to 1.0
    y[i] = 2.0f;  // Set all elements of y to 2.0
  }

  // ============================================================================
  // COMPUTATION PHASE
  // ============================================================================
  std::cout << "Starting CPU computation..." << std::endl;
  // Call the CPU addition function
  // This will execute sequentially on a single CPU core
  add(N, x, y);
  
  // Optional: Print sample results (commented out for performance)
  // Uncomment to verify first few elements
  /*
  std::cout << "Sample results:" << std::endl;
  for (int i = 0; i < 10 && i < N; i++) {
    std::cout << "y[" << i << "] = " << y[i] << std::endl;
  }
  */

  // ============================================================================
  // RESULT VALIDATION
  // ============================================================================
  // Verify computational correctness by checking all results
  // Expected: x[i] + y[i] = 1.0 + 2.0 = 3.0 for all elements
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
  std::cout << std::endl << "=== CPU BASELINE RESULTS ===" << std::endl;
  std::cout << "Elements processed: " << N << std::endl;
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
  // MEMORY CLEANUP
  // ============================================================================
  // Free allocated memory to prevent memory leaks
  delete [] x;
  delete [] y;
  
  std::cout << "CPU baseline computation complete." << std::endl;
  return 0;
}
