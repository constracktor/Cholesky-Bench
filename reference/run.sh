#!/bin/bash
#SBATCH --job-name=cholesky_reference
#SBATCH --output=logs/cholesky_reference_%j.out
#SBATCH --error=logs/cholesky_reference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=144:00:00
#SBATCH --exclusive
#
# Usage: run.sh
#
# Submit example:
#   sbatch run.sh

set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# Toolchain runtime selection
################################################################################
module load gcc/14.2.0

# Resolve directory where the script is located
SCRIPT_DIR="$(pwd)"

# OpenMP settings
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Make sure threaded MKL uses the OpenMP runtime if ENABLE_MKL=ON was used at
# build time.
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}

echo "Running with gcc runtime"

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_reference" \
  --loop 20 \
  --size_start 1024 \
  --size_stop 65536
