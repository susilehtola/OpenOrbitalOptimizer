find_package(Libxc CONFIG REQUIRED)
find_package(IntegratorXX CONFIG REQUIRED)

add_executable(
  atomtest-openorbopt
  src/atomtest.cpp
  )
target_link_libraries(
  atomtest-openorbopt
  PRIVATE
    ${ooo}::OpenOrbitalOptimizer
    Libxc::xc
    IntegratorXX::IntegratorXX
  )
add_test(  # Test #1
  NAME openorbopt/atomtest/build
  COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target atomtest-openorbopt
  )
set_tests_properties(
  openorbopt/atomtest/build
  PROPERTIES
    FIXTURES_SETUP OPENORBOPT_ATOMTEST_EXEC
  )
add_test(  # Test #2
  NAME openorbopt/atomtest/run1
  COMMAND $<TARGET_FILE:atomtest-openorbopt> --Z 8 --xfunc pbe --cfunc pbe --M 1 --sto 0 --basis "${CMAKE_CURRENT_SOURCE_DIR}/src/cc-pvdz.json"
  )
add_test(  # Test #3
  NAME openorbopt/atomtest/run2
  COMMAND $<TARGET_FILE:atomtest-openorbopt> --Z 8 --xfunc pbe --cfunc pbe --M 3 --sto 0 --basis "${CMAKE_CURRENT_SOURCE_DIR}/src/cc-pvdz.json"
  )
set_tests_properties(
  openorbopt/atomtest/run1
  openorbopt/atomtest/run2
  PROPERTIES
    FIXTURES_REQUIRED OPENORBOPT_ATOMTEST_EXEC
  )
