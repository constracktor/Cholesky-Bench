#!/bin/bash
#SBATCH --job-name=cholesky_std
#SBATCH --output=logs/cholesky_std_%j.out
#SBATCH --error=logs/cholesky_std_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=144:00:00
#SBATCH --exclusive
#
# Usage: run.sh
#
# Submit examples:
#   sbatch run.sh

# Load modules
module load gcc/14.2.0

# Resolve directory where the script is located
SCRIPT_DIR="$(pwd)"

# The parallel std algorithms (std::execution::par) take their worker count from
# the TBB backend; cap it to the allocated cores. std::async spawns its own OS
# threads per task and is not governed by this variable.
export TBB_NUM_THREADS=128

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_std" \
  --loop=20 \
  --size_start=65536 \
  --size_stop=65536 \
  --tiles_start=4 \
  --tiles_stop=1024
