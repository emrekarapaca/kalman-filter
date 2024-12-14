cmake_minimum_required(VERSION 3.28)
project(kalman_tracking)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)

add_executable(kalman_tracking main.cpp
        main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(kalman_tracking ${OpenCV_LIBS})
