cmake_minimum_required(VERSION 3.12)
project(RedLightTracking)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(red_light_tracking main.cpp)

target_include_directories(red_light_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(red_light_tracking PRIVATE ${OpenCV_LIBS})
