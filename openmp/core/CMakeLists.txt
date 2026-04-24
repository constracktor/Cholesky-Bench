set(SOURCE_FILES src/tile_generation.cpp src/functions.cpp
                 src/cholesky_factor.cpp src/adapter_cblas_fp64.cpp)

if(ENABLE_VALIDATION)
  list(APPEND SOURCE_FILES src/validate.cpp)
endif()

add_library(cholesky_core STATIC ${SOURCE_FILES})

# We'd like to consume it just as core under the GPRat namespace
set_property(TARGET cholesky_core PROPERTY EXPORT_NAME core)
add_library(Cholesky::core ALIAS cholesky_core)

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp)
target_sources(cholesky_core PRIVATE ${header_files})

# Link OpenMP libraries
target_link_libraries(cholesky_core PUBLIC OpenMP::OpenMP_CXX)

# Include directories
target_include_directories(
  cholesky_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

# Link BLAS (skipped entirely in no-op overhead-only builds).
if(DISABLE_COMPUTATION)
  target_compile_definitions(cholesky_core PUBLIC DISABLE_COMPUTATION)
elseif(ENABLE_MKL)
  # Link Intel oneMKL
  target_link_libraries(cholesky_core PUBLIC MKL::mkl_intel_lp64 MKL::mkl_core
                                             MKL::MKL MKL::mkl_sequential)
else()
  # Link OpenBLAS
  target_link_libraries(cholesky_core PUBLIC ${OpenBLAS_LIB})
  target_include_directories(cholesky_core PUBLIC ${OpenBLAS_INCLUDE_DIR})
endif()

if(ENABLE_MKL)
  target_compile_definitions(cholesky_core PUBLIC ENABLE_MKL)
endif()

target_compile_features(cholesky_core PUBLIC cxx_std_17)

set_property(TARGET cholesky_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # We need to manually install those into CMAKE_INSTALL_INCLUDEDIR. Below
  # install(TARGETS ...) only setups the paths for the exported targets.
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Development)

  install(
    TARGETS cholesky_core
    EXPORT CholeskyTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
