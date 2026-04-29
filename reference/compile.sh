#!/bin/bash
# Usage: compile.sh
#
# Builds the parallel-BLAS reference benchmark: a single threaded
# LAPACKE_dpotrf call on the full matrix, used as a baseline against the
# tiled OpenMP / HPX implementations. GCC only.
#
# CMake project options can be overridden via environment variables
# (defaults match the project's CMakeLists.txt defaults):
#   ENABLE_MKL          ON|OFF  (default OFF) - link threaded Intel oneMKL
#                                               instead of threaded OpenBLAS
#   ENABLE_VALIDATION   ON|OFF  (default OFF) - residual check after each
#                                               factorisation
#
# Examples:
#   ./compile.sh
#   ENABLE_MKL=ON ./compile.sh
#   ENABLE_VALIDATION=ON ./compile.sh
################################################################################
set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# CMake project options (env-var overridable; defaults match CMakeLists.txt)
################################################################################
: "${ENABLE_MKL:=OFF}"
: "${ENABLE_VALIDATION:=OFF}"

for var in ENABLE_MKL ENABLE_VALIDATION; do
  case "${!var}" in
  ON | OFF) ;;
  *)
    echo "Error: $var must be ON or OFF (got '${!var}')." >&2
    exit 1
    ;;
  esac
done

################################################################################
# Toolchain selection (gcc only)
################################################################################
select_toolchain() {
  module load gcc/14.2.0
  export CC=gcc
  export CXX=g++
}

################################################################################
# Configurations
#
# The reference benchmark uses *threaded* OpenBLAS / MKL — that is the whole
# point of this directory. The OpenMP and HPX builds, by contrast, pin the
# BLAS to its sequential variant because they parallelise at the tile level.
################################################################################
if command -v spack &>/dev/null; then
  echo "Spack command found. Loading libraries (gcc)"
  # Get current hostname
  HOSTNAME=$(hostname -s)

  if [[ "$HOSTNAME" == "ipvs-epyc1" || "$HOSTNAME" == "ipvs-epyc2" ]]; then
    # Compiler
    select_toolchain
    if [[ "$ENABLE_MKL" == "OFF" ]]; then
      # OpenBLAS built with OpenMP threading
      spack load openblas@0.3.28%gcc@14.2.0 threads=openmp ilp64=true
    fi

  elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
    # Compiler
    select_toolchain
    if [[ "$ENABLE_MKL" == "OFF" ]]; then
      # OpenBLAS built with OpenMP threading
      spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=openmp
    fi

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
echo "  ENABLE_MKL          = $ENABLE_MKL"
echo "  ENABLE_VALIDATION   = $ENABLE_VALIDATION"

cmake -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL="$ENABLE_MKL" \
  -DENABLE_VALIDATION="$ENABLE_VALIDATION" \
  ..
make -j VERBOSE=1
cd ..

# Launch Example
# OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores ./build/cholesky_reference --size_start 65536 --size_stop 65536 --loop 20
