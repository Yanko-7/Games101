cmake_minimum_required (VERSION 3.6)
project (Transformation)
add_executable (Transformation main.cpp)
target_link_libraries(Transformation Eigen3::Eigen)