cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")  # foil Ninja Debug on Windows

project(
  OpenOrbitalOptimizer
  VERSION 0.1.0
  DESCRIPTION
    "A library orbital optimization problems in quantum chemistry"
  HOMEPAGE_URL "https://github.com/susilehtola/OpenOrbitalOptimizer"
  LANGUAGES CXX
  )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(autocmake_safeguards)

# Version of libraries
set(SOVERSION 0.1.0)

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

find_package(Armadillo REQUIRED CONFIG)
if (NOT TARGET armadillo)
    message(FATAL "Require detecting an Armadillo target")
endif()
# [Mar 2025] Conda Windows target is wrong. Compensate with:
# * -D CMAKE_CXX_FLAGS="-I${CONDA_PREFIX}\\Library\\include"
# or use the block below, duplicated in ${ooo}Config.cmake
# Block below is cautious so if bad dir exists, may still need abov
get_property(_iid TARGET armadillo PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
if(NOT EXISTS "${_iid}")
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    if(DEFINED ENV{CONDA_PREFIX})
      message(STATUS "healing armadillo.INTERFACE_INCLUDE_DIRECTORIES from ${_iid} to $ENV{CONDA_PREFIX}\\Library\\include")
      set_property(TARGET armadillo PROPERTY INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}\\Library\\include")
    endif()
  endif()
endif()

target_link_libraries(
  libooo
  INTERFACE
    armadillo  # upstream target is not namespaced
  )

if (${ooo}_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
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
