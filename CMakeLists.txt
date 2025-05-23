cmake_minimum_required(VERSION 3.20)
project(VIRS)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use g++ explicitly
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# Define include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/args
)

# Recursively find all .cpp files
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Build the executable
add_executable(VIRS ${SRC_FILES})

# Optional: show all commands during build
set(CMAKE_VERBOSE_MAKEFILE ON)