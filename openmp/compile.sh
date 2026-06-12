#!/bin/bash
# Usage: compile.sh [compiler]
#   compiler: gcc | llvm   (default: gcc)
#
# CMake project options can be overridden via environment variables
# (defaults match the project's CMakeLists.txt defaults):
#   ENABLE_MKL              ON|OFF  (default OFF) - link Intel oneMKL instead of OpenBLAS
#   ENABLE_VALIDATION       ON|OFF  (default OFF) - residual check after each factorization
#   DISABLE_COMPUTATION     ON|OFF  (default OFF) - replace BLAS/tile-gen with no-ops
#   ENABLE_DYNAMIC_SCHEDULE ON|OFF  (default OFF) - schedule(dynamic,1) on trailing collapsed loop
#
# Examples:
#   ./compile.sh gcc
#   ENABLE_MKL=ON ./compile.sh gcc
#   ENABLE_VALIDATION=ON ./compile.sh gcc
#   ENABLE_DYNAMIC_SCHEDULE=ON ./compile.sh llvm
#   DISABLE_COMPUTATION=ON ./compile.sh llvm
################################################################################
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
# CMake project options (env-var overridable; defaults match CMakeLists.txt)
################################################################################
: "${ENABLE_MKL:=OFF}"
: "${ENABLE_VALIDATION:=OFF}"
: "${DISABLE_COMPUTATION:=OFF}"
: "${ENABLE_DYNAMIC_SCHEDULE:=OFF}"

for var in ENABLE_MKL ENABLE_VALIDATION DISABLE_COMPUTATION ENABLE_DYNAMIC_SCHEDULE; do
  case "${!var}" in
  ON | OFF) ;;
  *)
    echo "Error: $var must be ON or OFF (got '${!var}')." >&2
    exit 1
    ;;
  esac
done

# ENABLE_DYNAMIC_SCHEDULE=ON with GCC will fail at compile time
if [[ "$COMPILER" == "gcc" && "$ENABLE_DYNAMIC_SCHEDULE" == "ON" ]]; then
  echo "Error: ENABLE_DYNAMIC_SCHEDULE=ON is not supported with the gcc toolchain." >&2
  echo "       Use the llvm toolchain or set ENABLE_DYNAMIC_SCHEDULE=OFF." >&2
  exit 1
fi

################################################################################
# Toolchain selection
# ################################################################################
select_toolchain() {
  if [[ "$COMPILER" == "gcc" ]]; then
    module load gcc/14.2.0
    export CC=gcc
    export CXX=g++
  else
    spack load llvm@22.1.2
    export CC=clang
    export CXX=clang++
    export LD_LIBRARY_PATH=$(spack location -i llvm@22.1.2)/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
  fi
}

################################################################################
# Configurations
################################################################################
if command -v spack &>/dev/null; then
  echo "Spack command found. Loading libraries for compiler: $COMPILER"
  # Get current hostname
  HOSTNAME=$(hostname -s)

  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
    # Compiler
    select_toolchain
    # OpenBLAS
    spack load openblas@0.3.28%gcc@14.2.0 threads=none

  elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
    # Compiler
    select_toolchain
    # OpenBLAS
    spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=none

  else
    echo "Hostname is $HOSTNAME — no action taken."
  fi
else
  echo "Spack command not found. Exiting."
fi

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build

echo "CMake options:"
echo "  ENABLE_MKL              = $ENABLE_MKL"
echo "  ENABLE_VALIDATION       = $ENABLE_VALIDATION"
echo "  DISABLE_COMPUTATION     = $DISABLE_COMPUTATION"
echo "  ENABLE_DYNAMIC_SCHEDULE = $ENABLE_DYNAMIC_SCHEDULE"

cmake -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL="$ENABLE_MKL" \
  -DENABLE_VALIDATION="$ENABLE_VALIDATION" \
  -DDISABLE_COMPUTATION="$DISABLE_COMPUTATION" \
  -DENABLE_DYNAMIC_SCHEDULE="$ENABLE_DYNAMIC_SCHEDULE" \
  ..
make -j VERBOSE=1
cd ..

# Launch Example
# OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores OMP_MAX_TASK_PRIORITY=16 ./build/cholesky_openmp --size_start 65536 --size_stop 65536  --tiles_start 4 --tiles_stop 1024 --loop 1
