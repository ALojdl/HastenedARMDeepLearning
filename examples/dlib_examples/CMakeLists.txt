cmake_minimum_required(VERSION 2.8.12)
project(dlib_examples)
include(../../../dlib/dlib/cmake)

add_executable(andrej_test andrej_test.cpp)
target_link_libraries(andrej_test dlib::dlib)
