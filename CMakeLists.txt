cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")  # foil Ninja Debug on Windows


project(
  OpenOrbitalOptimizer
  VERSION 0.1.0
  DESCRIPTION
    "A library orbital optimization problems in quantum chemistry"
  HOMEPAGE_URL "https://github.com/susilehtola/openorbital"
  LANGUAGES CXX
  )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(autocmake_safeguards)

# Version of libraries
set(SOVERSION 0.1.0)

set(ooo OpenOrbitalOptimizer)  # Namespace

set(${ooo}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${ooo}"
    CACHE STRING "Directory to which CMake files are installed")
message(STATUS "Showing option ${ooo}_INSTALL_CMAKEDIR: ${${ooo}_INSTALL_CMAKEDIR}")


add_library(libooo INTERFACE)
set(ooo_TARGET_TYPE INTERFACE)

target_compile_features(
  libooo
  ${ooo_TARGET_TYPE}
    cxx_std_17
  )
target_include_directories(
  libooo
  ${ooo_TARGET_TYPE}
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/openorbital/include>
  )

install(
  DIRECTORY
    openorbital/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${ooo}
  FILES_MATCHING
    PATTERN "*.hpp*"
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
    #cmake/FindTargetLAPACK.cmake
    #cmake/FindTargetHDF5.cmake
  DESTINATION
    ${${ooo}_INSTALL_CMAKEDIR}
  )
