#!/bin/bash
# Usage: compile.sh
#
# Builds the parallel-BLAS reference benchmark: a single tile parallel
# LAPACKE_dpotrf call on the full matrix, used as a baseline against the
# tiled fork-join and tasking implementations.
#
# CMake project options can be overridden via environment variables
# (defaults match the project's CMakeLists.txt defaults):
#   ENABLE_MKL             ON|OFF  (default OFF) - link threaded Intel oneMKL
#                                                  instead of threaded OpenBLAS
#   ENABLE_PLASMA          ON|OFF  (default OFF) - also build the PLASMA
#                                                  plasma_dpotrf variant (extra
#                                                  'plasma' column in the output)
#   ENABLE_LAPACKE         ON|OFF  (default ON)  - run the LAPACKE_dpotrf
#                                                  reference mode at runtime
#   ENABLE_VALIDATION      ON|OFF  (default OFF) - residual check after each
#                                                  factorization
#
# Examples:
#   ./compile.sh
#   ENABLE_MKL=ON ./compile.sh
#   ENABLE_PLASMA=ON ./compile.sh
#   ENABLE_LAPACKE=OFF ENABLE_PLASMA=ON ./compile.sh
#   ENABLE_VALIDATION=ON ./compile.sh
################################################################################
set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# CMake project options (env-var overridable; defaults match CMakeLists.txt)
################################################################################
: "${ENABLE_MKL:=OFF}"
: "${ENABLE_PLASMA:=OFF}"
: "${ENABLE_LAPACKE:=ON}"
: "${ENABLE_VALIDATION:=OFF}"

for var in ENABLE_MKL ENABLE_PLASMA ENABLE_LAPACKE ENABLE_VALIDATION; do
  case "${!var}" in
  ON | OFF) ;;
  *)
    echo "Error: $var must be ON or OFF (got '${!var}')." >&2
    exit 1
    ;;
  esac
done

################################################################################
# Toolchain selection
################################################################################
select_toolchain() {
  module load gcc/14.2.0
  export CC=gcc
  export CXX=g++
}

################################################################################
# Configurations
#
# The reference benchmark uses *threaded* BLAS as they operate on a single tile
# and do not parallelize at the tile level.
################################################################################
if command -v spack &>/dev/null; then
  echo "Spack command found. Loading libraries."
  # Get current hostname
  HOSTNAME=$(hostname -s)

  if [[ "$HOSTNAME" == "ipvs-epyc1" || "$HOSTNAME" == "ipvs-epyc2" ]]; then
    # Compiler
    select_toolchain
    if [[ "$ENABLE_MKL" == "OFF" ]]; then
      # OpenBLAS built with OpenMP threading
      spack load openblas@0.3.28%gcc@14.2.0 threads=openmp ilp64=true
    fi
    if [[ "$ENABLE_PLASMA" == "ON" ]]; then
      spack load plasma%gcc@14.2.0 ^openblas@0.3.28%gcc@14.2.0 threads=openmp
    fi

  elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
    # Compiler
    select_toolchain
    if [[ "$ENABLE_MKL" == "OFF" ]]; then
      # OpenBLAS built with OpenMP threading
      spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=openmp ilp64=true
    fi
    if [[ "$ENABLE_PLASMA" == "ON" ]]; then
      spack load plasma%gcc@14.2.0 arch=linux-almalinux9-zen3 openblas@0.3.28%gcc@14.2.0 threads=openmp
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
echo "  ENABLE_MKL        = $ENABLE_MKL"
echo "  ENABLE_PLASMA     = $ENABLE_PLASMA"
echo "  ENABLE_LAPACKE    = $ENABLE_LAPACKE"
echo "  ENABLE_VALIDATION = $ENABLE_VALIDATION"

cmake -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL="$ENABLE_MKL" \
  -DENABLE_PLASMA="$ENABLE_PLASMA" \
  -DENABLE_LAPACKE="$ENABLE_LAPACKE" \
  -DENABLE_VALIDATION="$ENABLE_VALIDATION" \
  ..
make -j VERBOSE=1
cd ..

# Launch Example
# OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
# ./build/cholesky_reference --size_start 1024 --size_stop 65536 --loop 1
