# CS5379 GPU Programming Lecture - Build System
# ==============================================
# Builds all versions of vector addition (v0-v5)
# Usage: make [target]
#   make all    - Build all versions
#   make clean  - Remove executables
#   make help   - Show this help

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
CUDA_FLAGS = -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

# Source files and executables
SOURCES_CPP = add_v0.cpp
SOURCES_CU = add_v1.cu add_v2.cu add_v3.cu add_v4.cu add_v5.cu
EXECUTABLES = add_v0.exe add_v1.exe add_v2.exe add_v3.exe add_v4.exe add_v5.exe

# Default target
all: $(EXECUTABLES)
	@echo "✅ All versions compiled successfully!"
	@echo "Run './interactive_test.sh' or individual executables:"
	@echo "   ./add_v0.exe  # CPU baseline"
	@echo "   ./add_v5.exe  # Peak GPU performance"

# CPU version (C++)
add_v0.exe: add_v0.cpp
	@echo "🖥️  Compiling CPU version (v0)..."
	$(CXX) $(CXXFLAGS) $< -o $@

# CUDA versions
add_v%.exe: add_v%.cu cuda_error_check.h
	@echo "🚀 Compiling CUDA version (v$*)..."
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Individual targets for explicit building
app0: add_v0.exe
app1: add_v1.exe  
app2: add_v2.exe
app3: add_v3.exe
app4: add_v4.exe
app5: add_v5.exe

# Cleanup
clean:
	@echo "🧹 Cleaning up executables..."
	rm -f $(EXECUTABLES)
	@echo "✅ Clean complete!"

# Help information
help:
	@echo "CS5379 GPU Programming Lecture - Build System"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all     - Build all versions (default)"
	@echo "  app0    - Build CPU version only"
	@echo "  app1-5  - Build specific CUDA version"
	@echo "  clean   - Remove all executables"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make           # Build everything"
	@echo "  make app3      # Build only v3"
	@echo "  make clean all # Clean and rebuild"

# Utility targets
check-cuda:
	@echo "🔍 Checking CUDA installation..."
	@which nvcc > /dev/null && echo "✅ NVCC found: $$(nvcc --version | grep release)" || echo "❌ NVCC not found"
	@which nvidia-smi > /dev/null && echo "✅ NVIDIA driver found" || echo "❌ NVIDIA driver not found"

info:
	@echo "📊 Build Configuration:"
	@echo "  C++ Compiler: $(CXX)"
	@echo "  CUDA Compiler: $(NVCC)"
	@echo "  C++ Flags: $(CXXFLAGS)"
	@echo "  CUDA Flags: $(CUDA_FLAGS)"
	@echo "  Executables: $(EXECUTABLES)"

# Phony targets
.PHONY: all clean help check-cuda info app0 app1 app2 app3 app4 app5
