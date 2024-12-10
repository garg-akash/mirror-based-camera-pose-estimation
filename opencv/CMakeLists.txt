cmake_minimum_required(VERSION 3.10)
project(demo_project)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target
add_executable(demo demo.cc sub_solveP3P.h tnm.h)

# Link OpenCV libraries
target_link_libraries(demo ${OpenCV_LIBS})

# Custom target for running the executable
# add_custom_target(run
#     COMMAND demo
#     DEPENDS demo
#     WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
# )
