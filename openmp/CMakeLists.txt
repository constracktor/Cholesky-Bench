cmake_minimum_required(VERSION 3.23)
project(cholesky_opemp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# What to build?
option(BUILD_CORE "Build the core library" ON)
option(ENABLE_MKL "Enable Intel oneMKL support" OFF)
option(ENABLE_VALIDATION
       "Compute ||A - L*L^T||_F / ||A||_F after each factorization (off by default)"
       OFF)
option(DISABLE_COMPUTATION
       "Replace all BLAS/LAPACK calls and tile generation with no-ops; keeps wrappers, loops and task graph intact so that OpenMP scheduling overhead can be measured in isolation"
       OFF)
option(ENABLE_DYNAMIC_SCHEDULE
       "Use schedule(dynamic, 1) on the trailing-update worksharing loops in for_collapse and for_naive. OFF by default (static schedule) so GCC builds compile out of the box; libgomp through 14.x cannot combine dynamic scheduling with collapse on a non-rectangular nest. Turn ON for LLVM builds where the dynamic schedule is supported and gives better load balancing."
       OFF)
option(ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

if(ENABLE_VALIDATION AND DISABLE_COMPUTATION)
  message(FATAL_ERROR
          "ENABLE_VALIDATION and DISABLE_COMPUTATION are mutually exclusive: "
          "residual validation needs a real factorization to check against.")
endif()

if(ENABLE_FORMAT_TARGETS)
  find_package(format QUIET)
  if(NOT format_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      format
      GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
      GIT_TAG v1.8.1
      QUIET)
    FetchContent_MakeAvailable(format)
  endif()
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

  find_package(OpenMP REQUIRED)

  add_subdirectory(core)

  # Add the executable
  add_executable(cholesky_openmp main.cpp)

  # Link the libraries
  target_link_libraries(cholesky_openmp PUBLIC Cholesky::core OpenMP::OpenMP_CXX)

  if(ENABLE_VALIDATION)
    target_compile_definitions(cholesky_openmp PRIVATE ENABLE_VALIDATION)
  endif()

  if(DISABLE_COMPUTATION)
    target_compile_definitions(cholesky_openmp PRIVATE DISABLE_COMPUTATION)
  endif()
endif()
