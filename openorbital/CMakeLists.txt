# Create the orbopt library
add_library(openorbital src/scfsolver.cpp)

add_executable(atomtest src/atomtest.cpp)
target_link_libraries(atomtest openorbital)


include_directories("/home/work/optim/include")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
