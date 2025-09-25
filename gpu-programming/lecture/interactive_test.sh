#!/bin/bash

# ============================================
# Interactive Performance Testing Script
# CS5379 GPU Programming Lecture
# ============================================
# This script is designed to run interactively on h100 partition
# Use: srun --partition=h100 --gpus-per-node=1 --pty bash
# Then: ./interactive_test.sh
# ============================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS=("v0" "v1" "v2" "v3" "v4" "v5")

print_header() {
  echo ""
  echo -e "${CYAN}=============================================${NC}"
  echo -e "${CYAN} $1${NC}"
  echo -e "${CYAN}=============================================${NC}"
}

check_gpu_access() {
  print_header "Checking GPU Access"
  
  if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi not found. Are you on a GPU node?${NC}"
    return 1
  fi
  
  if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ Cannot access GPU. Are you in an interactive session with GPU?${NC}"
    echo -e "${YELLOW}💡 Try: srun --partition=h100 --gpus-per-node=1 --pty bash${NC}"
    return 1
  fi
  
  echo -e "${GREEN}✅ GPU Access confirmed${NC}"
  nvidia-smi --query-gpu=name,memory.total --format=csv
}

load_modules() {
  print_header "Loading Modules"
  
  echo -e "${BLUE}📦 Loading GCC and CUDA modules...${NC}"
  module load gcc/14.2.0 cuda
  
  echo -e "${GREEN}✅ Modules loaded${NC}"
  echo -e "${BLUE}GCC Version: $(gcc --version | head -1)${NC}"
  echo -e "${BLUE}NVCC Version: $(nvcc --version | tail -1)${NC}"
}

compile_all() {
  print_header "Compiling All Versions"
  
  cd "$SCRIPT_DIR"
  
  # Clean previous builds
  echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
  make clean 2>/dev/null || rm -f *.exe
  
  echo -e "${BLUE}🔨 Compiling all versions...${NC}"
  make all
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All versions compiled successfully${NC}"
    ls -la *.exe
  else
    echo -e "${RED}❌ Compilation failed${NC}"
    return 1
  fi
}

run_performance_test() {
  print_header "Running Performance Tests"
  
  local results_file="interactive_results_$(date +%Y%m%d_%H%M%S).txt"
  
  echo -e "${BLUE}📊 Running performance analysis...${NC}"
  echo -e "${BLUE}Results will be saved to: $results_file${NC}"
  
  {
    echo "=============================================="
    echo "Interactive GPU Performance Analysis"
    echo "=============================================="
    echo "Date: $(date)"
    echo "Node: $(hostname)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "=============================================="
    echo ""
  } | tee "$results_file"
  
  for version in "${VERSIONS[@]}"; do
    local exe_file="add_${version}.exe"
    
    if [ ! -f "$exe_file" ]; then
      echo -e "${RED}❌ Executable $exe_file not found${NC}"
      continue
    fi
    
    if [ "$version" = "v0" ]; then
      echo -e "${YELLOW}🖥️  Testing CPU Version (C++)${NC}"
      echo "🖥️  Testing CPU Version (C++):" | tee -a "$results_file"
      echo "   - Sequential processing on CPU" | tee -a "$results_file"
    else
      echo -e "${GREEN}🚀 Testing CUDA Version $version${NC}"
      echo "🚀 Testing CUDA Version $version:" | tee -a "$results_file"
      case $version in
          v1) echo "   - 1 block, 1 thread (very slow, sequential on GPU)" | tee -a "$results_file" ;;
          v2) echo "   - 1 block, 256 threads (parallel within block)" | tee -a "$results_file" ;;
          v3) echo "   - Many blocks, 256 threads each (full GPU utilization)" | tee -a "$results_file" ;;
          v4) echo "   - Many blocks + GPU-based initialization" | tee -a "$results_file" ;;
          v5) echo "   - Many blocks + GPU initialization + memory prefetching" | tee -a "$results_file" ;;
      esac
    fi
    
    echo "   ⏱️  Timing:" | tee -a "$results_file"
    { time ./"$exe_file"; } 2>&1 | tee -a "$results_file"
    echo "" | tee -a "$results_file"
    
    # Run profiling for CUDA versions
    if [ "$version" != "v0" ]; then
      echo -e "${PURPLE}🔍 Running profiling for $version...${NC}"
      echo "   📊 Nsight Systems Profile:" | tee -a "$results_file"
      nsys nvprof ./"$exe_file" 2>&1 | tee -a "$results_file" || echo "   ⚠️  Profiling failed or not available" | tee -a "$results_file"
      echo "" | tee -a "$results_file"
    fi
  done
  
  {
    echo "=============================================="
    echo "Performance Summary:"
    echo "=============================================="
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
    echo "=============================================="
  } | tee -a "$results_file"
  
  echo -e "${GREEN}✅ Performance analysis complete!${NC}"
  echo -e "${BLUE}📄 Results saved to: $results_file${NC}"
}

show_help() {
  echo "Interactive Performance Testing Script"
  echo ""
  echo "This script compiles and tests all CUDA versions interactively."
  echo "It's designed to run on h100 partition with GPU access."
  echo ""
  echo "Usage:"
  echo "  1. Get interactive session: srun --partition=h100 --gpus-per-node=1 --pty bash"
  echo "  2. Run this script: ./interactive_test.sh"
  echo ""
  echo "Options:"
  echo "  -h, --help     Show this help message"
  echo "  --compile-only Only compile, don't run tests"
  echo "  --test-only    Only run tests (skip compilation)"
}

main() {
  local compile_only=false
  local test_only=false
  
  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        show_help
        exit 0
        ;;
      --compile-only)
        compile_only=true
        shift
        ;;
      --test-only)
        test_only=true
        shift
        ;;
      *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
    esac
  done
  
  print_header "Interactive GPU Performance Testing"
  echo -e "${GREEN}🚀 Starting interactive performance analysis...${NC}"
  echo -e "${BLUE}📅 Date: $(date)${NC}"
  
  # Check GPU access
  if ! check_gpu_access; then
    exit 1
  fi
  
  # Load modules
  load_modules
  
  # Compile if needed
  if [ "$test_only" = false ]; then
    compile_all
  fi
  
  # Run tests if needed
  if [ "$compile_only" = false ]; then
    run_performance_test
  fi
  
  echo -e "${GREEN}🎉 Interactive analysis complete!${NC}"
}

main "$@"