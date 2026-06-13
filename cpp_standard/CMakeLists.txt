cmake_minimum_required(VERSION 3.23)
project(cholesky_std CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# What to build?
option(BUILD_CORE "Build the core library" ON)
option(ENABLE_MKL "Enable Intel oneMKL support" OFF)
option(
  ENABLE_VALIDATION
  "Compute ||A - L*L^T||_F / ||A||_F after each factorization (off by default)"
  OFF)
option(
  DISABLE_COMPUTATION
  "Replace all BLAS/LAPACK calls and tile generation with no-ops; keeps the std::future task graph and parallel loops intact so scheduling overhead can be measured in isolation"
  OFF)
if(ENABLE_VALIDATION AND DISABLE_COMPUTATION)
  message(
    FATAL_ERROR
      "ENABLE_VALIDATION and DISABLE_COMPUTATION are mutually exclusive: "
      "residual validation needs a real factorization to check against.")
endif()

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # Our installs follow the standard GNU directory layout. This include needs to
  # come first since we need the CMAKE_INSTALL_* in the CMakeLists.txt of each
  # target.
  include(GNUInstallDirs)
endif()

if(BUILD_CORE)
  if(DISABLE_COMPUTATION)
    message(STATUS "DISABLE_COMPUTATION=ON - skipping BLAS/LAPACK discovery; "
                   "wrappers will be compiled as no-ops.")
  elseif(ENABLE_MKL)
    # Try to find Intel oneMKL
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "sequential")
    find_package(MKL CONFIG REQUIRED)

    if(MKL_FOUND)
      message(STATUS "Intel oneMKL Library found")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  else()
    # Try to find OpenBLAS
    find_library(OpenBLAS_LIB NAMES openblas REQUIRED)

    if(OpenBLAS_LIB)
      message(STATUS "OpenBLAS Library found at ${OpenBLAS_LIB}")
      find_path(
        OpenBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES openblas)
      if(NOT OpenBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "OpenBLAS include directory not found")
      endif()

      message(STATUS "OpenBLAS include dir: ${OpenBLAS_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  endif()

  # std::async tasking needs a threading library; the parallel std::for_each
  # backend in libstdc++ is implemented on top of Intel TBB and must be linked
  # explicitly. (libc++/MSVC do not require TBB; there the find below is a
  # best-effort no-op.)
  find_package(Threads REQUIRED)
  find_package(TBB QUIET)
  if(TBB_FOUND)
    message(STATUS "TBB found - parallel std algorithms will use it")
  else()
    message(
      STATUS
        "TBB not found - parallel std::for_each may fall back to sequential "
        "execution (libstdc++ requires TBB for std::execution::par).")
  endif()

  add_subdirectory(core)

  # Add the executable
  add_executable(cholesky_std main.cpp)

  # Link the libraries
  target_link_libraries(cholesky_std PUBLIC Cholesky::core)

  if(ENABLE_VALIDATION)
    target_compile_definitions(cholesky_std PRIVATE ENABLE_VALIDATION)
  endif()

  if(DISABLE_COMPUTATION)
    target_compile_definitions(cholesky_std PRIVATE DISABLE_COMPUTATION)
  endif()
endif()
