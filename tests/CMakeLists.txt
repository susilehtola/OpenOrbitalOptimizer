find_package(Libxc CONFIG REQUIRED)
find_package(IntegratorXX CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# The atom tester uses (Torb,Tbase) = (double,double)
add_executable(
  openorbopt-atomtest
  atomtest.cpp
  )
target_link_libraries(
  openorbopt-atomtest
  PRIVATE
    ${ooo}::OpenOrbitalOptimizer
    armadillo  # atomicsolver.hpp + the SCFSolver bridge in this test
    Libxc::xc
    IntegratorXX::IntegratorXX
    nlohmann_json::nlohmann_json
  )
add_test(  # Test #1
  NAME openorbopt/atomtest/build
  COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-atomtest
  )
set_tests_properties(
  openorbopt/atomtest/build
  PROPERTIES
    FIXTURES_SETUP OPENORBOPT_ATOMTEST_EXEC
  )
# On Windows, the run1/run2 tests give `Exit code 0xc0000374` = STATUS_HEAP_CORRUPTION with MinGW (Gnu) compilers. Use clang-cl instead.
add_test(  # Test #2
  NAME openorbopt/atomtest/run1
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 8 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --M 1 --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json"
  )
add_test(  # Test #3
  NAME openorbopt/atomtest/run2
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 8 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --M 3 --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json"
  )
set_tests_properties(
  openorbopt/atomtest/run1
  openorbopt/atomtest/run2
  PROPERTIES
    FIXTURES_REQUIRED OPENORBOPT_ATOMTEST_EXEC
  )

  # Instantiations of (Torb,Tbase) = (float,float), (std::complex<float>,float), and (std::complex<double>,double)
  add_executable(
    openorbopt-instantiation-float-float
    EXCLUDE_FROM_ALL
    float_float.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-float-float
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_executable(
    openorbopt-instantiation-cplxfloat-float
    EXCLUDE_FROM_ALL
    cplxfloat_float.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-cplxfloat-float
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_executable(
    openorbopt-instantiation-cplxdouble-double
    EXCLUDE_FROM_ALL
    cplxdouble_double.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-cplxdouble-double
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(  # Test #4
    NAME openorbopt/float-float/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-float-float
  )
  add_test(  # Test #5
    NAME openorbopt/cplxfloat-float/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-cplxfloat-float
  )
  add_test(  # Test #6
    NAME openorbopt/cplxdouble-double/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-cplxdouble-double
  )

  # Quad-precision (_Float128) instantiations. These require C++23 because
  # libstdc++'s std::sqrt/exp/log/... overloads for _Float128 are gated by
  # __cpp_lib_extended_float. Excluded from the default build; opt-in via
  # the ctest targets.
  add_executable(
    openorbopt-instantiation-quad-real
    EXCLUDE_FROM_ALL
    quad_real.cpp
  )
  target_compile_features(openorbopt-instantiation-quad-real PRIVATE cxx_std_23)
  target_link_libraries(
    openorbopt-instantiation-quad-real
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_executable(
    openorbopt-instantiation-quad-complex
    EXCLUDE_FROM_ALL
    quad_complex.cpp
  )
  target_compile_features(openorbopt-instantiation-quad-complex PRIVATE cxx_std_23)
  target_link_libraries(
    openorbopt-instantiation-quad-complex
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(  # Test #7
    NAME openorbopt/quad-real/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-quad-real
  )
  add_test(  # Test #8
    NAME openorbopt/quad-complex/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-quad-complex
  )
