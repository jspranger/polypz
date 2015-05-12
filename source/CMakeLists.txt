cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Polyps)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(Polyps main.cpp)
target_link_libraries(Polyps ITKCommon ITKStatistics)