cmake_minimum_required(VERSION 3.23)
project(cholesky_reference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# What to build?
option(BUILD_CORE "Build the core library" ON)
option(ENABLE_MKL "Enable Intel oneMKL support (threaded)" OFF)
option(
  ENABLE_PLASMA
  "Build the PLASMA tiled-Cholesky reference variant in addition to the LAPACKE_dpotrf one"
  OFF)
option(
  ENABLE_LAPACKE
  "Run the LAPACKE_dpotrf reference mode at runtime (on by default). Linking is unchanged either way (PLASMA and validation still need cblas/lapacke)."
  ON)
option(
  ENABLE_VALIDATION
  "Compute ||A - L*L^T||_F / ||A||_F after each factorization (off by default)"
  OFF)
option(ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

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
  include(GNUInstallDirs)
endif()

if(BUILD_CORE)
  if(ENABLE_MKL)
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "intel_thread")
    find_package(MKL CONFIG REQUIRED)

    if(MKL_FOUND)
      message(STATUS "Intel oneMKL Library found (threaded: ${MKL_THREADING})")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  else()
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

  if(ENABLE_PLASMA)
    find_path(PLASMA_INCLUDE_DIR plasma.h)
    if(NOT PLASMA_INCLUDE_DIR)
      message(FATAL_ERROR "ENABLE_PLASMA=ON but plasma.h was not found")
    endif()
    find_library(PLASMA_LIB NAMES plasma REQUIRED)
    find_library(PLASMA_CORE_BLAS_LIB NAMES coreblas plasma_core_blas)
    message(STATUS "PLASMA include dir: ${PLASMA_INCLUDE_DIR}")
    message(STATUS "PLASMA library: ${PLASMA_LIB}")
    if(PLASMA_CORE_BLAS_LIB)
      message(STATUS "PLASMA coreblas library: ${PLASMA_CORE_BLAS_LIB}")
    endif()
  endif()

  add_subdirectory(core)

  # Add the executable
  add_executable(cholesky_reference main.cpp)

  # Link the libraries
  target_link_libraries(cholesky_reference PUBLIC Cholesky::core
                                                  OpenMP::OpenMP_CXX)

  if(ENABLE_VALIDATION)
    target_compile_definitions(cholesky_reference PRIVATE ENABLE_VALIDATION)
  endif()

  if(ENABLE_PLASMA)
    target_compile_definitions(cholesky_reference PRIVATE ENABLE_PLASMA)
    target_include_directories(cholesky_reference PRIVATE ${PLASMA_INCLUDE_DIR})
  endif()

  if(ENABLE_LAPACKE)
    target_compile_definitions(cholesky_reference PRIVATE ENABLE_LAPACKE)
  endif()
endif()
