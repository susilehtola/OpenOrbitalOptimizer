# Create the orbopt library

add_executable(atomtest src/atomtest.cpp)
target_link_libraries(atomtest xc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
