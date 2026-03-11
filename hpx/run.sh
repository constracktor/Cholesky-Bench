#!/bin/bash
#SBATCH --job-name=cholesky_hpx
#SBATCH --output=logs/cholesky_hpx_%j.out
#SBATCH --error=logs/cholesky_hpx_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --exclusive

# Load modules if needed
module load gcc/14.2.0

# Get directory where this script resides
SCRIPT_DIR="$(pwd)"

# Ensure the job runs from the script directory
cd "$SCRIPT_DIR"

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_hpx" \
    --hpx:threads=128 \
    --loop=20 \
    --size_start=512 \
    --size_stop=65536 \
    --tiles_start=4 \
    --tiles_stop=256
