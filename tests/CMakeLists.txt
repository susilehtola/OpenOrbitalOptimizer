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
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 8 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --M 1 --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json" --reference-energy -74.9164866611 --energy-tolerance 1e-8
  )
add_test(  # Test #3
  NAME openorbopt/atomtest/run2
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 8 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --M 3 --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json" --reference-energy -74.9722875549 --energy-tolerance 1e-8
  )
# Transition metals. These are the cases where the occupations of a
# fractionally filled shell are free to settle anywhere along a very
# flat valley -- iron shares its valence between 4s and 3d -- and
# before the occupations were held to their stationarity conditions
# the same command reproducibly gave two different answers 2e-6 Eh
# apart, both reporting convergence. There was nothing stable to
# assert until then, which is why they are only now in CI.
#
# They are run without DIIS. The extrapolation converges these cases
# on its own and the orbital-rotation step barely runs, so a mix
# containing it would not exercise what these tests are here for.
#
# Each takes three to five minutes, comparable to the instantiation
# builds below.
add_test(  # Test #4
  NAME openorbopt/atomtest/iron-unrestricted
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 26 --M 5 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json" --methods "ODA + ARH" --max-gradient 1e-5 --reference-energy -1263.4350147950 --energy-tolerance 1e-7 --max-fermi-level-error 1e-5
  )
# The restricted atom asserts its gradient explicitly rather than
# deferring to the solver's own verdict, and is held to a tighter
# Fermi-level residual.
#
# Settling the occupations perturbs the orbitals, and winning that back
# is worth about g^2/2H -- some 1e-12 at the gradient reached here,
# three orders under what the Fock builder reproduces. No line search
# can pay for it, so where the gradient ends up is decided by the
# arithmetic rather than by the algorithm, and it does not reproduce
# between machines: 5.0e-7 and 1.1e-6 on the two kinds of core here,
# 2.26e-6 on CI. A threshold set from the first two duly failed on the
# third.
#
# 1e-5 is ten times the convergence threshold, which this case holds
# with room to spare, while --max-fermi-level-error and
# --reference-energy keep the physics under test. Demanding convergence
# outright would only mean reporting the unsettled occupations instead:
# this case used to pass with a Fermi-level error of 0.13.
add_test(  # Test #5
  NAME openorbopt/atomtest/iron-restricted
  COMMAND $<TARGET_FILE:openorbopt-atomtest> --Z 26 --M 0 --xfunc GGA_X_PBE --cfunc GGA_C_PBE --basis "${CMAKE_CURRENT_SOURCE_DIR}/cc-pvdz.json" --methods "ODA + ARH" --max-gradient 1e-5 --reference-energy -1263.3342431303 --energy-tolerance 1e-6 --max-fermi-level-error 1e-5
  )
set_tests_properties(
  openorbopt/atomtest/run1
  openorbopt/atomtest/run2
  openorbopt/atomtest/iron-unrestricted
  openorbopt/atomtest/iron-restricted
  PROPERTIES
    FIXTURES_REQUIRED OPENORBOPT_ATOMTEST_EXEC
  )

  # Runtime coverage of the SCFSolver settings façade: catalog
  # round-trip and unknown-key rejection.
  add_executable(
    openorbopt-settings-roundtrip
    settings_roundtrip.cpp
  )
  target_link_libraries(
    openorbopt-settings-roundtrip
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(
    NAME openorbopt/settings/roundtrip
    COMMAND $<TARGET_FILE:openorbopt-settings-roundtrip>
  )

  # Aufbau cleanup on a Fermi level degenerate across more than two
  # orbitals, where the occupation polytope is not a line. The atomic
  # driver cannot reach this, its blocks holding whole l shells.
  add_executable(
    openorbopt-degenerate-occupations
    degenerate_occupations.cpp
  )
  target_link_libraries(
    openorbopt-degenerate-occupations
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(
    NAME openorbopt/degenerate-occupations
    COMMAND $<TARGET_FILE:openorbopt-degenerate-occupations>
  )

  # Conventions of the ARH curvature pairs, checked against
  # numerically differentiated gradients on a model whose two particle
  # types are coupled.
  add_executable(
    openorbopt-arh-hessian
    arh_hessian.cpp
  )
  target_link_libraries(
    openorbopt-arh-hessian
    PRIVATE
    ${ooo}::OpenOrbitalOptimizer
  )
  add_test(
    NAME openorbopt/arh-hessian
    COMMAND $<TARGET_FILE:openorbopt-arh-hessian>
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

  # Armadillo compatibility shim (opt-in for downstream callers that still
  # carry arma::-shaped data). The core library does not link Armadillo, so
  # this target is only added when an Armadillo installation is present.
  find_package(Armadillo CONFIG QUIET)
  if (Armadillo_FOUND AND TARGET armadillo)
    add_executable(
      openorbopt-armadillo-compat
      EXCLUDE_FROM_ALL
      armadillo_compat.cpp
    )
    target_link_libraries(
      openorbopt-armadillo-compat
      PRIVATE
      ${ooo}::OpenOrbitalOptimizer
      armadillo
    )
    add_test(  # Test #7
      NAME openorbopt/armadillo-compat/build
      COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target openorbopt-armadillo-compat
    )
  else()
    message(STATUS "Skipping Armadillo compatibility shim test: Armadillo not found")
  endif()

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
    # Give the atom driver a quad-precision path too. The solver runs in
    # _Float128 while the Fock build stays double, which brackets where a
    # numerical effect originates -- see restricted_scf_quad. Needs C++23
    # for the libstdc++ _Float128 math overloads, so it is only switched
    # on where those exist.
    target_compile_features(openorbopt-atomtest PRIVATE cxx_std_23)
    target_compile_definitions(openorbopt-atomtest PRIVATE OPENORBOPT_QUAD)

    # libquadmath is GCC's runtime for __float128/_Float128 math intrinsics.
    # libstdc++'s std::sqrt/exp/log/... for _Float128 emit calls into it on
    # some toolchains and link directly into the program on others. Link it
    # when available; harmless if all the math has been inlined.
    find_library(QUADMATH_LIBRARY NAMES quadmath)
    if (QUADMATH_LIBRARY)
      target_link_libraries(openorbopt-atomtest PRIVATE ${QUADMATH_LIBRARY})
    endif()

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

# ====  Fully numerical tests, via HelFEM  =====================================
#
# HelFEM's finite-element drivers use this library for their orbital
# optimization, which makes them the natural place to test it without a
# basis set in the way: a converged HelFEM atom has no basis-set error,
# so what is left is the solver. The fractional-occupation work in
# particular is easier to judge there -- on iron at M = 0 the 4s/3d
# split differs from its cc-pVDZ value in the third decimal, while the
# stationarity conditions settle it to eight, so the basis and the
# algorithm are cleanly separated.
#
# Built as an ExternalProject rather than add_subdirectory. HelFEM
# brings some fifty tests of its own and a good many targets, and none
# of that belongs in this project's suite; this way only the driver
# binary crosses the boundary.
if(${ooo}_BUILD_HELFEM_TESTS)
  include(ExternalProject)

  if(${ooo}_HELFEM_SOURCE_DIR)
    set(_helfem_source SOURCE_DIR "${${ooo}_HELFEM_SOURCE_DIR}")
    message(STATUS "HelFEM tests: building from ${${ooo}_HELFEM_SOURCE_DIR}")
  else()
    set(_helfem_source
        GIT_REPOSITORY https://github.com/susilehtola/HelFEM.git
        GIT_TAG        ${${ooo}_HELFEM_GIT_TAG})
    message(STATUS "HelFEM tests: fetching ${${ooo}_HELFEM_GIT_TAG} from GitHub")
  endif()

  # OpenOrbitalOptimizer_DIR points at this build tree, which the
  # export(EXPORT) in the top-level CMakeLists makes consumable in
  # place. HelFEM's own find_package then succeeds and its FetchContent
  # fallback -- the one pinned to a fixed commit -- never fires, so the
  # driver is compiled against these headers.
  ExternalProject_Add(helfem_numerical
    ${_helfem_source}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DOpenOrbitalOptimizer_DIR=${PROJECT_BINARY_DIR}
      -DHELFEM_BINARIES=ON
      -DHELFEM_PYTHON=OFF
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target gensap
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    USES_TERMINAL_BUILD TRUE
  )
  ExternalProject_Get_Property(helfem_numerical BINARY_DIR)
  set(HELFEM_GENSAP "${BINARY_DIR}/src/gensap")

  add_test(
    NAME openorbopt/helfem/build
    COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target helfem_numerical
  )
  set_tests_properties(openorbopt/helfem/build PROPERTIES TIMEOUT 3600)

  # Iron at M = 0, spin-restricted, fully numerical: the same case the
  # basis-set tests run, with the basis-set error removed. Without DIIS,
  # for the reason given above the transition-metal tests.
  #
  # The functional is given with a dash, which is how HelFEM separates
  # exchange from correlation.
  add_test(
    NAME openorbopt/helfem/iron-restricted
    COMMAND ${HELFEM_GENSAP} --Z Fe --M 0 --restricted 1
            --lmax 3 --nelem 10 --nnodes 8
            --method gga_x_pbe-gga_c_pbe
            --scfmethods "ODA + ARH"
  )
  # Matched to seven decimals. The energy reproduces exactly across a
  # local build and a clean clone; what does scatter, by the same
  # trajectory noise the basis-set tests show, is the last digit or two
  # of the occupations and the stationarity residual -- 1.09114 against
  # 1.09115 in the 4s, 2.8e-8 against 1.3e-8 in the residual -- so those
  # are not pinned. Asserting only that something converged would pass
  # on a wrong answer.
  set_tests_properties(
    openorbopt/helfem/iron-restricted
    PROPERTIES
      DEPENDS openorbopt/helfem/build
      TIMEOUT 600
      PASS_REGULAR_EXPRESSION "Converged to energy -1263\\.3357152"
  )
endif()
