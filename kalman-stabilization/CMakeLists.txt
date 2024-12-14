cmake_minimum_required(VERSION 3.28)
project(kalman_stabilization)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(kalman_stabilization main.cpp)

target_link_libraries(kalman_stabilization ${OpenCV_LIBS})
