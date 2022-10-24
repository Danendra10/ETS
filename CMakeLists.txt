cmake_minimum_required(VERSION 3.0)
project(ETS)

add_compile_options(-std=c++17)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    linpng++
    )

add_executable(ETS main.cpp)
add_dependencies(ETS ${OpenCV_LIBS})
target_link_libraries(ETS ${OpenCV_LIBS})