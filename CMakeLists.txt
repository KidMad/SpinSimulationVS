cmake_minimum_required(VERSION 3.21)
project(SpinSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
find_package(OpenMP REQUIRED)
add_executable(SpinSimulation "main.cpp" "tools.cpp" "tools.h")
target_link_libraries(SpinSimulation PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(SpinSimulation PUBLIC ../eigen-3.4.0)
