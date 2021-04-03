
cmake_minimum_required(VERSION 3.19)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

project(mandelbrot)

add_executable(mandelbrot mandelbrot.cpp)


set(CMAKE_CXX_FLAGS "-Wall -Wextra -Ofast -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "")