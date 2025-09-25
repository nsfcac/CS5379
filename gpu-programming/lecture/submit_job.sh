#!/bin/bash
#SBATCH --job-name=CS5379_eraider_ID
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=1

# ========================================
# Program Version Selection
# ========================================
# Get version from command line argument or default to v5
PROGRAM_VERSION="${1:-v5}"

# Validate the version argument
case "$PROGRAM_VERSION" in
    v0|v1|v2|v3|v4|v5)
        echo "✅ Valid version selected: ${PROGRAM_VERSION}"
        ;;
    *)
        echo "❌ Invalid version: ${PROGRAM_VERSION}"
        echo "Usage: sbatch submit_job.sh [version]"
        echo "Valid versions: v0, v1, v2, v3, v4, v5"
        echo "Example: sbatch submit_job.sh v3"
        exit 1
        ;;
esac

# Use compatible GCC version (14.2.0 works better than 15.1.0)
module load gcc/14.2.0 cuda

echo "========================================="
echo "CUDA Vector Addition - Version ${PROGRAM_VERSION}"
echo "Usage: sbatch submit_job.sh [version]"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Output files: CS5379_eraider_ID_${SLURM_JOB_ID}_${PROGRAM_VERSION}.out/.err"
echo "========================================="

# Set source file and executable based on version
if [ "$PROGRAM_VERSION" = "v0" ]; then
    SRC_FILE="add_v0.cpp"
    EXE_FILE="add_v0.exe"
    COMPILER="g++"
    echo "Compiling CPU version (C++)..."
else
    SRC_FILE="add_${PROGRAM_VERSION}.cu"
    EXE_FILE="add_${PROGRAM_VERSION}.exe"
    COMPILER="nvcc"
    echo "Compiling CUDA version ${PROGRAM_VERSION}..."
fi

# Compile the selected version
echo "Source: ${SRC_FILE}"
echo "Executable: ${EXE_FILE}"
echo "Compiler: ${COMPILER}"
echo ""

if [ "$PROGRAM_VERSION" = "v0" ]; then
    # Compile CPU version
    ${COMPILER} ${SRC_FILE} -o ${EXE_FILE}
else
    # Compile CUDA version
    ${COMPILER} ${SRC_FILE} -o ${EXE_FILE}
fi

# Check compilation status
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
    echo ""
    
    # Run the program with timing
    echo "⏱️  Running ${EXE_FILE} with timing measurements..."
    echo "----------------------------------------"
    
    if [ "$PROGRAM_VERSION" = "v0" ]; then
        # Time the CPU version multiple times for better statistics
        echo "🖥️  CPU Baseline Performance (C++ version):"
        echo "Running 3 iterations for average timing..."
        echo ""
        
        total_time=0
        for i in {1..3}; do
            echo "Run ${i}/3:"
            # Use 'time' command to measure execution time (redirect stderr to stdout)
            exec_time=$( { time ./${EXE_FILE} > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
            echo "  Execution time: ${exec_time}"
        done
        
        echo ""
        echo "🎯 Running final timed execution with output:"
        # Capture time output properly by redirecting stderr to stdout
        { time ./${EXE_FILE}; } 2>&1
        echo ""
        echo "📊 CPU Performance Summary:"
        echo "  - Pure CPU execution (single-threaded)"
        echo "  - No GPU acceleration"
        echo "  - Baseline for comparison"
        
    else
        # Time CUDA versions  
        echo "🚀 CUDA Performance (Version ${PROGRAM_VERSION}):"
        echo "Running timed execution:"
        # Capture time output properly by redirecting stderr to stdout
        { time ./${EXE_FILE}; } 2>&1
    fi
    
    echo "----------------------------------------"
    echo ""
    
    # Profile CUDA versions only (skip CPU version)
    if [ "$PROGRAM_VERSION" != "v0" ]; then
        echo "📊 Profiling with Nsight Systems..."
        nsys nvprof ./${EXE_FILE}
        echo "✅ Profiling completed"
        echo ""
        echo "Generated files:"
        ls -la *.nsys-rep *.sqlite 2>/dev/null | tail -2 || echo "No new profiling files found"
    else
        echo "ℹ️  Skipping GPU profiling for CPU version"
        echo "💡 CPU timing data available above for performance comparison"
    fi
else
    echo "❌ Compilation failed"
    echo "Check error messages above"
fi

echo ""
echo "========================================="
echo "Job completed for version ${PROGRAM_VERSION}"

# Rename output files to include version number
if [ -n "${SLURM_JOB_ID}" ]; then
    OLD_OUT="CS5379_eraider_ID.${SLURM_JOB_ID}.out"
    OLD_ERR="CS5379_eraider_ID.${SLURM_JOB_ID}.err" 
    NEW_OUT="CS5379_eraider_ID_${SLURM_JOB_ID}_${PROGRAM_VERSION}.out"
    NEW_ERR="CS5379_eraider_ID_${SLURM_JOB_ID}_${PROGRAM_VERSION}.err"
    
    echo ""
    echo "📁 Renaming output files to include version:"
    echo "  ${OLD_OUT} -> ${NEW_OUT}"
    echo "  ${OLD_ERR} -> ${NEW_ERR}"
    
    # Perform the rename using a post-processing approach
    # Note: This happens at the end of job execution
    if [ -f "${OLD_OUT}" ]; then
        cp "${OLD_OUT}" "${NEW_OUT}" 2>/dev/null || echo "Could not rename output file"
        rm -f "${OLD_OUT}" 2>/dev/null || echo "Could not remove old output file"
    fi
    if [ -f "${OLD_ERR}" ]; then
        cp "${OLD_ERR}" "${NEW_ERR}" 2>/dev/null || echo "Could not rename error file"
        rm -f "${OLD_ERR}" 2>/dev/null || echo "Could not remove old error file"
    fi
fi
echo "========================================="
