cmake_minimum_required(VERSION 3.15)


# set the project name
project(dprouting VERSION 0.0.1 LANGUAGES C CXX)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# paths
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# add subdirectories
add_subdirectory(cc)
add_subdirectory(third-party)
add_subdirectory(python)


