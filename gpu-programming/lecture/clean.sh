#!/bin/bash

# clean.sh - Clean up CUDA profiling and SLURM output files
# Author: Mert Side
# Date: September 24, 2025

echo "========================================="
echo "CUDA Profiling & SLURM Cleanup Script"
echo "Date: $(date)"
echo "========================================="
echo ""

# Function to safely remove files with confirmation
cleanup_files() {
  local pattern="$1"
  local description="$2"
  local files=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null)
  
  if [ -n "$files" ]; then
    echo "Found $description files:"
    echo "$files" | sed 's/^/  - /'
    local count=$(echo "$files" | wc -l)
    echo "  Total: $count files"
    echo ""
  else
    echo "No $description files found."
  fi
}

# Function to actually remove files
remove_files() {
  local pattern="$1"
  local description="$2"
  local files=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null)
  
  if [ -n "$files" ]; then
    echo "$files" | xargs rm -f
    local count=$(echo "$files" | wc -l)
    echo "✅ Removed $count $description files"
  else
    echo "ℹ️  No $description files to remove"
  fi
}

# Show what will be cleaned
echo "Scanning for files to clean..."
echo ""

cleanup_files "*.sqlite" "SQLite database"
cleanup_files "*.nsys-rep" "Nsight Systems report"
cleanup_files "*.out" "SLURM output"
cleanup_files "*.err" "SLURM error"
cleanup_files "*.exe" "compiled executable"

echo "========================================="

# Ask for confirmation
read -p "Do you want to remove all these files? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo ""
  echo "Cleaning up files..."
  echo ""
  
  # Remove each type of file
  remove_files "*.sqlite" "SQLite database"
  remove_files "*.nsys-rep" "Nsight Systems report"
  remove_files "*.out" "SLURM output"
  remove_files "*.err" "SLURM error"
  remove_files "*.exe" "compiled executable"
  
  echo ""
  echo "========================================="
  echo "✅ Cleanup completed successfully!"
  echo "========================================="
else
  echo ""
  echo "❌ Cleanup cancelled by user."
  echo "========================================="
fi

echo ""
echo "Remaining files in directory:"
ls -la *.cu *.cpp *.sh *.md 2>/dev/null || echo "  (Source and script files preserved)"
echo ""