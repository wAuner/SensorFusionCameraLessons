cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)
set(OpenCV_DIR ~/opencv-4.1.1/installation/lib/cmake/opencv4)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executables for exercise
add_executable (cornerness_harris src/cornerness_harris.cpp)
target_link_libraries (cornerness_harris ${OpenCV_LIBRARIES})

add_executable (solution ../solution/cornerness_harris.cpp)
target_link_libraries (solution ${OpenCV_LIBRARIES})
