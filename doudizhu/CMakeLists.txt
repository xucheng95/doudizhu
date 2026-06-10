cmake_minimum_required(VERSION 3.16)
project(doudizhu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Library
add_library(doudizhu
    src/cards.cpp
    src/game.cpp
    src/env.cpp
)
target_include_directories(doudizhu PUBLIC include)

# Tests (optional, manual cmake -DBUILD_TESTS=ON)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_executable(doudizhu_test
        test/cards_test.cpp
        test/game_test.cpp
    )
    target_link_libraries(doudizhu_test doudizhu GTest::GTest)
endif()

# Pybind11 binding
option(BUILD_PYTHON "Build python bindings" ON)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(doudizhu_cpp pybind/bindings.cpp)
    target_link_libraries(doudizhu_cpp PRIVATE doudizhu)
    target_include_directories(doudizhu_cpp PRIVATE include)
endif()
