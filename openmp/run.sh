#!/bin/bash
#SBATCH --job-name=cholesky_openmp
#SBATCH --output=logs/cholesky_openmp_%j.out
#SBATCH --error=logs/cholesky_openmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=144:00:00
#SBATCH --exclusive
#
# Usage: run.sh [compiler]
#   compiler: gcc | llvm   (default: gcc; must match how the binary was built)
#
# Submit examples:
#   sbatch run.sh            # uses gcc runtime
#   sbatch run.sh gcc
#   sbatch run.sh llvm

set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# Argument parsing
################################################################################
COMPILER="${1:-gcc}"
case "$COMPILER" in
gcc | llvm) ;;
*)
  echo "Error: unknown compiler '$COMPILER' (expected 'gcc' or 'llvm')." >&2
  echo "Usage: $0 [gcc|llvm]" >&2
  exit 1
  ;;
esac

################################################################################
# Toolchain runtime selection
################################################################################
select_runtime() {
  if [[ "$COMPILER" == "gcc" ]]; then
    module load gcc/14.2.0
  else
    spack load llvm@22.1.2
    export LD_LIBRARY_PATH=$(spack location -i llvm@22.1.2)/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
  fi
}
select_runtime

# Resolve directory where the script is located
SCRIPT_DIR="$(pwd)"

# OpenMP settings
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Required for the task_prio variant (priority() is otherwise capped to 0).
export OMP_MAX_TASK_PRIORITY=16

echo "Running with $COMPILER runtime"

# Run executable
srun --cpu-bind=cores "$SCRIPT_DIR/build/cholesky_openmp" \
  --loop 20 \
  --size_start 65536 \
  --size_stop 65536 \
  --tiles_start 4 \
  --tiles_stop 1024
