cmake_minimum_required(VERSION 3.12)
project (liborbopt)
set(PROJECT_TYPE C CXX)
set(C_STANDARD 99)
set(CXX_STANDARD 17)

# Enable Fortran for LAPACK
enable_language (Fortran)

# Set version
set(VERSION 0.1.0)
# Version of libraries
set(SOVERSION 0.1.0)

find_package(OpenMP)
if(OPENMP_FOUND)
  # Add flags to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # Set OMP flags for preprocessor
  set(PPROC_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

# Find LAPACK
find_package(LAPACK REQUIRED)
link_libraries(${LAPACK_LIBRARIES})

# Find Armadillo
option(ARMA_DONT_USE_WRAPPER "Do not use Armadillo wrapper" ON)
#find_package(Armadillo REQUIRED)
#include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(".")
include_directories("/home/work/armadillo/include")
include_directories("/home/work/IntegratorXX/include")
if(ARMA_DONT_USE_WRAPPER)
  add_definitions(-DARMA_DONT_USE_WRAPPER)
else()
  link_libraries(${ARMADILLO_LIBRARIES})
endif()

# Descend into subdirectory
add_subdirectory(openorbital)
