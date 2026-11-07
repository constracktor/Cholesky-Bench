#!/bin/bash
#SBATCH --job-name=cholesky_starpu
#SBATCH --output=logs/cholesky_starpu_%j.out
#SBATCH --error=logs/cholesky_starpu_%j.err
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

# StarPU scheduler (lws = locality work stealing, the StarPU default; alternatives: eager, ws, dmda, ...)
export STARPU_SCHED=lws
# Suppress StarPU warnings/banner on stderr
export STARPU_SILENT=1
# For n_tiles >= 512 the number of submitted tasks (~n^3/6) becomes huge; throttle submission to bound memory:
# export STARPU_LIMIT_MAX_SUBMITTED_TASKS=100000
# export STARPU_LIMIT_MIN_SUBMITTED_TASKS=80000
# main.cpp reserves a core for the submitting main thread by default when >= 8 threads are used
# (set STARPU_MAIN_THREAD_BIND=0 to disable).

# Resolve directory where the script is located
SCRIPT_DIR="$(pwd)"

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_starpu" \
  --threads 128 \
  --loop 20 \
  --size_start 65536 \
  --size_stop 65536 \
  --tiles_start 4 \
  --tiles_stop 1024
