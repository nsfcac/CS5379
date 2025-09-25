#!/bin/bash

echo "============================================================"
echo "GPU Kernel Performance Analysis"
echo "============================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "============================================================"
echo ""

# Test each version and record timing
for version in v0 v1 v2 v3 v4 v5; do
  echo "============================================================"
  if [ "$version" = "v0" ]; then      
    echo "🖥️  Testing CPU Version (C++):"
    echo "   - Sequential processing on CPU"
  else
    echo "🚀 Testing CUDA Version $version:"
    case $version in
      v1) echo "   - 1 block, 1 thread (very slow, sequential on GPU)" ;;
      v2) echo "   - 1 block, 256 threads (parallel within block)" ;;
      v3) echo "   - Many blocks, 256 threads each (full GPU utilization)" ;;
      v4) echo "   - Many blocks + GPU-based initialization" ;;
      v5) echo "   - Many blocks + GPU initialization + memory prefetching" ;;
    esac
  fi
  
  echo -n "   Result: "
  echo ""
  echo "   ⏱️  Timing:"
  time ./add_${version}.exe
  echo ""
done

echo "============================================================"
echo "Performance Summary:"
echo "============================================================"
echo "✅ v0 (CPU): Baseline sequential performance"
echo "⚠️  v1 (1 thread): Slowest GPU version - no parallelism"
echo "✅ v2 (1 block): Better - uses 256 threads in parallel"
echo "🚀 v3 (many blocks): Best basic GPU performance"
echo "🚀 v4 (GPU init): Optimized memory initialization"
echo "🚀 v5 (prefetch): Best performance with memory prefetching"
echo ""
echo "Key Insights:"
echo "- GPU parallelism requires many threads across many blocks"
echo "- Memory management strategy significantly affects performance"
echo "- Error checking is essential for debugging GPU code"
echo "============================================================"