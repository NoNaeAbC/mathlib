cmake_minimum_required(VERSION 3.17)
project(mathlib)

find_package(Lua REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(mathlib main.cpp)
add_executable(test_complex testComplex.cpp)
add_executable(mathliblua aml_lua_binding.cpp aml_lua_binding.h testlua.cpp)

target_link_libraries(mathliblua lua)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (UNIX)
    set(CMAKE_C_COMPILER "gcc")# gcc clang
    set(CMAKE_CXX_COMPILER "g++")# g++ clang++


    message(STATUS ${CMAKE_SYSTEM_PROCESSOR})

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(CMAKE_CXX_FLAGS "-Wall -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -mcpu=cortex-a72 -ffast-math -lmpfr -lgmp")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=cortex-a72 -ffast-math -lmpfr -lgmp")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(CMAKE_CXX_FLAGS "-Wall -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -ffast-math -march=native -lmpfr -lgmp")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -lmpfr -lgmp -flto")
    else ()

        set(CMAKE_CXX_FLAGS "-Wall -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -ffast-math")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto")
    endif ()

endif ()
