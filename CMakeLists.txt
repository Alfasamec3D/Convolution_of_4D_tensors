cmake_minimum_required(VERSION 3.13)

project("Triangles" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(dim STATIC lib/dim.cpp)

target_include_directories(dim PUBLIC "include/dim")


add_library(tensor INTERFACE)

target_include_directories(tensor INTERFACE "include/tensor")

target_link_libraries(tensor INTERFACE dim)


add_library(gemm STATIC lib/gemm.cpp)

target_include_directories(gemm PUBLIC "include/gemm")

target_link_libraries(gemm PUBLIC tensor)


add_library(convolution STATIC lib/convolution.cpp)

target_include_directories(convolution PUBLIC "include/convolution")

target_link_libraries(convolution PUBLIC tensor PRIVATE gemm PRIVATE dim)


add_executable(main main.cpp)

target_link_libraries(main PRIVATE convolution)


option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    add_subdirectory(tests)
endif()