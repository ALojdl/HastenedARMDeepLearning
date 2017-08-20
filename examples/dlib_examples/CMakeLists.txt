cmake_minimum_required(VERSION 2.8.12)
project(dlib_examples)
include(../../../dlib/dlib/cmake)

add_executable(small_neural_network small_neural_network.cpp)
target_link_libraries(small_neural_network dlib::dlib)
