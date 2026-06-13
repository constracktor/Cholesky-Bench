#!/bin/bash
# Usage: compile.sh
#
# Builds the pure-standard-library Cholesky benchmark (std::future tasking +
# parallel std::for_each fork-join). No HPX runtime is involved.
#
# CMake project options can be overridden via environment variables
# (defaults match the project's CMakeLists.txt defaults):
#   ENABLE_MKL          ON|OFF  (default OFF) - link Intel oneMKL instead of OpenBLAS
#   ENABLE_VALIDATION   ON|OFF  (default OFF) - residual check after each factorization
#   DISABLE_COMPUTATION ON|OFF  (default OFF) - replace BLAS/tile-gen with no-ops
#
# Examples:
#   ./compile.sh
#   ENABLE_MKL=ON ./compile.sh
#   ENABLE_VALIDATION=ON ./compile.sh
#   DISABLE_COMPUTATION=ON ./compile.sh
################################################################################
set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# CMake project options (env-var overridable; defaults match CMakeLists.txt)
################################################################################
: "${ENABLE_MKL:=OFF}"
: "${ENABLE_VALIDATION:=OFF}"
: "${DISABLE_COMPUTATION:=OFF}"

for var in ENABLE_MKL ENABLE_VALIDATION DISABLE_COMPUTATION; do
  case "${!var}" in
  ON | OFF) ;;
  *)
    echo "Error: $var must be ON or OFF (got '${!var}')." >&2
    exit 1
    ;;
  esac
done

if [[ "$ENABLE_VALIDATION" == "ON" && "$DISABLE_COMPUTATION" == "ON" ]]; then
  echo "Error: ENABLE_VALIDATION and DISABLE_COMPUTATION are mutually exclusive:" >&2
  echo "       residual validation needs a real factorization to check against." >&2
  exit 1
fi

################################################################################
# Configurations
#
# The standard-library build only needs a C++20 compiler, a sequential BLAS,
# and (for libstdc++) Intel TBB to back std::execution::par. On the project's
# clusters these come from Spack/modules; elsewhere the find_package() calls in
# CMake pick up system installs.
################################################################################
if command -v spack &>/dev/null; then
  echo "Spack command found. Loading libraries."
  # Get current hostname
  HOSTNAME=$(hostname -s)

  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
    module load gcc/14.2.0
    export CC=gcc
    export CXX=g++
    spack load openblas@0.3.28%gcc@14.2.0 threads=none
    spack load intel-tbb%gcc@14.2.0 2>/dev/null || spack load tbb%gcc@14.2.0 2>/dev/null || true

  elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
    module load gcc/14.2.0
    export CC=gcc
    export CXX=g++
    spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=none
    spack load intel-tbb%gcc@14.2.0 2>/dev/null || spack load tbb%gcc@14.2.0 2>/dev/null || true

  else
    echo "Hostname is $HOSTNAME — no action taken (relying on system toolchain)."
  fi
else
  echo "Spack command not found. Relying on the system toolchain / BLAS / TBB."
fi

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build

echo "CMake options:"
echo "  ENABLE_MKL          = $ENABLE_MKL"
echo "  ENABLE_VALIDATION   = $ENABLE_VALIDATION"
echo "  DISABLE_COMPUTATION = $DISABLE_COMPUTATION"

cmake -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL="$ENABLE_MKL" \
  -DENABLE_VALIDATION="$ENABLE_VALIDATION" \
  -DDISABLE_COMPUTATION="$DISABLE_COMPUTATION" \
  ..
make -j
cd ..

# Example
# ./build/cholesky_std \
# --loop=1 \
# --size_start=2048 \
# --size_stop=2048 \
# --tiles_start=16 \
# --tiles_stop=16
