find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ext _ext.cpp)
target_link_libraries(_ext PRIVATE ${ooo}::OpenOrbitalOptimizer)
set_target_properties(
  _ext
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openorbital
  )

# Copy the Python facade alongside the extension so the build tree
# is directly importable as ``openorbital``.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/openorbital/__init__.py
  ${CMAKE_CURRENT_BINARY_DIR}/openorbital/__init__.py
  COPYONLY
  )

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
