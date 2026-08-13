cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")  # foil Ninja Debug on Windows

project(
  OpenOrbitalOptimizer
  VERSION 0.4.0
  DESCRIPTION
    "A library orbital optimization problems in quantum chemistry"
  HOMEPAGE_URL "https://github.com/susilehtola/OpenOrbitalOptimizer"
  LANGUAGES CXX
  )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(autocmake_safeguards)

# Version of libraries
set(SOVERSION 0.4.0)

set(ooo OpenOrbitalOptimizer)  # Namespace

# TODO: Remove when requiring cmake >= 3.21  # from LecrisUT/CMake-Template
if (NOT DEFINED ${ooo}_IS_TOP_LEVEL)
    if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
        set(PROJECT_IS_TOP_LEVEL ON)
    else ()
        set(PROJECT_IS_TOP_LEVEL OFF)
    endif ()
    set(${ooo}_IS_TOP_LEVEL ${PROJECT_IS_TOP_LEVEL})
endif ()

# ====  Options  ================================================================

# there are paths that work for both windows and unix but they repeat ${ooo} :-(
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    set(${ooo}_INSTALL_CMAKEDIR "${ooo}/CMake"
        CACHE STRING "Directory to which CMake files are installed")
else()
    set(${ooo}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${ooo}"
        CACHE STRING "Directory to which CMake files are installed")
endif()
message(STATUS "Showing option ${ooo}_INSTALL_CMAKEDIR: ${${ooo}_INSTALL_CMAKEDIR}")

option(${ooo}_BUILD_TESTING "Build test-suite" ${PROJECT_IS_TOP_LEVEL})
message(STATUS "Showing option ${ooo}_BUILD_TESTING: ${${ooo}_BUILD_TESTING}")

option(${ooo}_BUILD_PYTHON "Build Python bindings via pybind11" OFF)
message(STATUS "Showing option ${ooo}_BUILD_PYTHON: ${${ooo}_BUILD_PYTHON}")

# Fully numerical regression tests, run through HelFEM's finite-element
# drivers. Off by default: HelFEM wants BLAS/LAPACK, HDF5, libxc and
# OpenMP on top of what the rest of the suite needs, and building it
# takes several minutes -- more than the whole of the rest of the
# suite.
#
# HelFEM is built from source against *this* tree rather than against
# an installed OpenOrbitalOptimizer, and that is the point of the
# exercise. HelFEM pins the version it fetches to an explicit commit,
# quite deliberately, so a HelFEM built the ordinary way links whatever
# snapshot that pin names -- and a test suite run against it would
# report on a library nobody is editing. Ask a stock HelFEM for the ARH
# step and it answers "Unknown method 'arh'".
option(${ooo}_BUILD_HELFEM_TESTS "Build fully numerical tests via HelFEM" OFF)
message(STATUS "Showing option ${ooo}_BUILD_HELFEM_TESTS: ${${ooo}_BUILD_HELFEM_TESTS}")

set(${ooo}_HELFEM_SOURCE_DIR "" CACHE PATH
    "Local HelFEM checkout to build the numerical tests from; fetched from GitHub when empty")
# Pinned to a commit rather than to master, for the reason HelFEM gives
# for pinning us: with a floating branch two builds of identical sources
# can link different dependencies and produce different numbers, with
# nothing in the history to explain it, which makes bisection
# meaningless. Bump this deliberately, in its own commit.
set(${ooo}_HELFEM_GIT_TAG "b01dacb1b744cfc1e7f0b345eaf7441d67e5e7f6" CACHE STRING
    "HelFEM commit to fetch when no local checkout is given")

# ====  Build  ==================================================================
add_library(libooo INTERFACE)

target_compile_features(
  libooo
  INTERFACE
    cxx_std_17
  )
target_include_directories(
  libooo
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
if(MSVC)
    # MSVC does not include <cmath> constants, unless _USE_MATH_DEFINES is defined.
    target_compile_definitions(libooo INTERFACE _USE_MATH_DEFINES)
    # Set the exception handling model (allows "throw"; if using add_compile_options form, set this *before* adding targets)
    target_compile_options(libooo INTERFACE "/EHsc")
endif()

add_library(${ooo}::OpenOrbitalOptimizer ALIAS libooo)

find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(
  libooo
  INTERFACE
    Eigen3::Eigen
  )


if (${ooo}_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if (${ooo}_BUILD_PYTHON)
    add_subdirectory(python)
endif()

set(export_properties
  "${ooo}_VERSION"
  )

set_target_properties (
  libooo
  PROPERTIES
    EXPORT_NAME "${ooo}"
    ${ooo}_VERSION ${${ooo}_VERSION}
  )
set_property(TARGET libooo APPEND PROPERTY EXPORT_PROPERTIES "${export_properties}")

# ====  Export & Install  =======================================================
install(
  DIRECTORY
    openorbitaloptimizer
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.hpp*"
  )

install(
  TARGETS
    libooo
  EXPORT
    ho_interface
  RUNTIME DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION
    ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  EXPORT
    ho_interface
  FILE
    "${PROJECT_NAME}Targets.cmake"
  NAMESPACE
    "${PROJECT_NAME}::"
  DESTINATION
    ${${ooo}_INSTALL_CMAKEDIR}
  )

# Make the build tree consumable in place, not just the install tree.
# install(EXPORT) writes the targets file only when the project is
# installed, so -DOpenOrbitalOptimizer_DIR=<builddir> finds the
# generated Config but then fails on the missing Targets include. A
# consumer that wants to build against a working tree -- rather than
# against whatever snapshot was last installed -- needs this, and for a
# header-only library the build-tree interface points straight at the
# sources, which is exactly what such a consumer is asking for.
export(
  EXPORT
    ho_interface
  FILE
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
  NAMESPACE
    "${PROJECT_NAME}::"
  )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "cmake/${ooo}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${ooo}Config.cmake"
  INSTALL_DESTINATION ${${ooo}_INSTALL_CMAKEDIR}
  )

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${ooo}ConfigVersion.cmake"
  VERSION ${${ooo}_VERSION}
  COMPATIBILITY SameMinorVersion
  )

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${ooo}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${ooo}ConfigVersion.cmake
  DESTINATION
    ${${ooo}_INSTALL_CMAKEDIR}
  )
