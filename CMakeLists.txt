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

set(${ooo}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${ooo}"
    CACHE STRING "Directory to which CMake files are installed")
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

add_library(${ooo}::OpenOrbitalOptimizer ALIAS libooo)

find_package(Armadillo REQUIRED)
target_link_libraries(
  libooo
  INTERFACE
    armadillo  # upstream target is not namespaced
  )
message("ARMADILLO ${ARMADILLO_INCLUDE_DIRS} ${ARMADILLO_LIBRARIES}")

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
