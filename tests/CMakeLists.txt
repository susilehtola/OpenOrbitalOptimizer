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

  # Instantiations of SCFSolver<Tbase, IsComplex> for
  # <float, false>, <float, true>, and <double, true>.
  # The <double, false> path is exercised by the atomtest target above.
  add_executable(
    openorbopt-instantiation-float-real
    EXCLUDE_FROM_ALL
    float_real.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-float-real
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_executable(
    openorbopt-instantiation-float-complex
    EXCLUDE_FROM_ALL
    float_complex.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-float-complex
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_executable(
    openorbopt-instantiation-double-complex
    EXCLUDE_FROM_ALL
    double_complex.cpp
  )
  target_link_libraries(
    openorbopt-instantiation-double-complex
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(  # Test #4
    NAME openorbopt/float-real/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-float-real
  )
  add_test(  # Test #5
    NAME openorbopt/float-complex/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-float-complex
  )
  add_test(  # Test #6
    NAME openorbopt/double-complex/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-double-complex
  )

  # Quad-precision (_Float128) instantiations. Require both the
  # _Float128 language type (GCC extension, plus a recent libstdc++ that
  # has the std::sqrt/... overloads behind __cpp_lib_extended_float) and
  # C++23. Skipped on compilers (Clang, MSVC, icpx) and platforms (arm64
  # Apple) that lack one of the two.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-std=c++23")
  check_cxx_source_compiles("
    #include <cmath>
    int main() { _Float128 x = 2; return (int)std::sqrt(x); }
  " OPENORBOPT_HAS_FLOAT128)
  unset(CMAKE_REQUIRED_FLAGS)

  if (OPENORBOPT_HAS_FLOAT128)
    # libquadmath is GCC's runtime for __float128/_Float128 math intrinsics.
    # libstdc++'s std::sqrt/exp/log/... for _Float128 emit calls into it on
    # some toolchains and link directly into the program on others. Link it
    # when available; harmless if all the math has been inlined.
    find_library(QUADMATH_LIBRARY NAMES quadmath)

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
    if (QUADMATH_LIBRARY)
      target_link_libraries(openorbopt-instantiation-quad-real PRIVATE ${QUADMATH_LIBRARY})
    endif()
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
    if (QUADMATH_LIBRARY)
      target_link_libraries(openorbopt-instantiation-quad-complex PRIVATE ${QUADMATH_LIBRARY})
    endif()
    add_test(  # Test #7
      NAME openorbopt/quad-real/build
      COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-quad-real
    )
    add_test(  # Test #8
      NAME openorbopt/quad-complex/build
      COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-instantiation-quad-complex
    )
  else()
    message(STATUS "Skipping quad-precision (_Float128) tests: compiler does not provide std::sqrt(_Float128) under -std=c++23")
  endif()
