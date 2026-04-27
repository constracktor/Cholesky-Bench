#!/bin/bash
# Usage: compile.sh
#
# CMake project options can be overridden via environment variables
# (defaults match the project's CMakeLists.txt defaults):
#   ENABLE_VALIDATION   ON|OFF  (default OFF) - residual check after each factorization
#   DISABLE_COMPUTATION ON|OFF  (default OFF) - replace BLAS/tile-gen with no-ops
#
# Examples:
#   ./compile.sh
#   ENABLE_VALIDATION=ON ./compile.sh
#   DISABLE_COMPUTATION=ON ./compile.sh
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# CMake project options (env-var overridable; defaults match CMakeLists.txt)
################################################################################
: "${ENABLE_VALIDATION:=OFF}"
: "${DISABLE_COMPUTATION:=OFF}"

for var in ENABLE_VALIDATION DISABLE_COMPUTATION; do
    case "${!var}" in
        ON|OFF) ;;
        *)
            echo "Error: $var must be ON or OFF (got '${!var}')." >&2
            exit 1
            ;;
    esac
done

# Caught at CMake configure time too (FATAL_ERROR), but reject earlier
# here with a clearer message so the user does not wait through the spack
# loads only to fail at configure.
if [[ "$ENABLE_VALIDATION" == "ON" && "$DISABLE_COMPUTATION" == "ON" ]]; then
    echo "Error: ENABLE_VALIDATION and DISABLE_COMPUTATION are mutually exclusive:" >&2
    echo "       residual validation needs a real factorization to check against." >&2
    exit 1
fi

################################################################################
# Configurations
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found. Loading libraries."
    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	module load gcc/14.2.0
	export CC=gcc
	export CXX=g++
	spack load hpx@1.11.0%gcc@14.2.0 malloc=jemalloc
	spack load openblas@0.3.28%gcc@14.2.0 threads=none

    elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
	module load gcc/14.2.0
	spack load hpx@1.11.0%gcc@14.2.0 arch=linux-almalinux9-zen3
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
echo "  ENABLE_VALIDATION   = $ENABLE_VALIDATION"
echo "  DISABLE_COMPUTATION = $DISABLE_COMPUTATION"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_VALIDATION="$ENABLE_VALIDATION" \
      -DDISABLE_COMPUTATION="$DISABLE_COMPUTATION" \
      -DHPX_IGNORE_BOOST_COMPATIBILITY=ON ..
make -j

cd ..
./build/cholesky_hpx \
    --hpx:threads=128 \
    --loop=1 \
    --size_start=65536 \
    --size_stop=65536 \
    --tiles_start=64 \
    --tiles_stop=64
