# Use the modern FindPython module rather than the deprecated
# FindPythonLibs / FindPythonInterp that pybind11 falls back to by
# default. Silences the CMP0148 deprecation warning on CMake >= 3.27
# and works with current and future pybind11 releases.
#
# When called inside an activated conda environment, point CMake's
# FindPython at the conda interpreter. CMake's default "highest
# version on the system" search picks /usr/bin/python3.12 on multi-
# arch Debian / Ubuntu CI runners (whose pyconfig.h is a stub that
# only resolves when the system python-dev package is installed)
# and on macOS picks the system python (which has the wrong ABI for
# the conda-built .so), breaking either the build or the test step.
#
# Python_ROOT_DIR is only a hint and gets merged with the default
# search list; Python_EXECUTABLE is the documented hard override and
# is what we need here.
if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED Python_EXECUTABLE)
  if(WIN32)
    set(_oo_conda_python "$ENV{CONDA_PREFIX}/python.exe")
  else()
    set(_oo_conda_python "$ENV{CONDA_PREFIX}/bin/python")
  endif()
  if(EXISTS "${_oo_conda_python}")
    set(Python_EXECUTABLE "${_oo_conda_python}")
  endif()
  unset(_oo_conda_python)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ext _ext.cpp)
target_link_libraries(_ext PRIVATE ${ooo}::OpenOrbitalOptimizer)
set_target_properties(
  _ext
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openorbital
  )

# Copy the Python facade alongside the extension so the build tree
# is directly importable as ``openorbital``. Mirror every .py file
# under the source ``openorbital/`` package -- including subpackages
# such as ``openorbital.pyscf`` -- into the build tree at the same
# relative path.
file(GLOB_RECURSE _ooo_py_files
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/openorbital
     ${CMAKE_CURRENT_SOURCE_DIR}/openorbital/*.py)
foreach(_pyfile IN LISTS _ooo_py_files)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/openorbital/${_pyfile}
    ${CMAKE_CURRENT_BINARY_DIR}/openorbital/${_pyfile}
    COPYONLY
    )
endforeach()

# Optional install target: install both the extension and the Python
# facade into the same site-packages directory.
include(GNUInstallDirs)
if (Python_SITELIB)
  install(
    TARGETS _ext
    LIBRARY DESTINATION ${Python_SITELIB}/openorbital
    )
  install(
    DIRECTORY openorbital/
    DESTINATION ${Python_SITELIB}/openorbital
    FILES_MATCHING PATTERN "*.py"
    )
endif()
