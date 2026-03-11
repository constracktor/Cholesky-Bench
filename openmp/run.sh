#!/bin/bash
#SBATCH --job-name=cholesky_openmp
#SBATCH --output=logs/cholesky_openmp_%j.out
#SBATCH --error=logs/cholesky_openmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --exclusive

# Resolve directory where the script is located
SCRIPT_DIR="$(pwd)"

# OpenMP settings
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_openmp" \
    --loop 20 \
    --size_start 512 \
    --size_stop 65536 \
    --tiles_start 4 \
    --tiles_stop 256
