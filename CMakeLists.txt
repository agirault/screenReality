cmake_minimum_required(VERSION 2.8)
PROJECT( openglcv_project )

find_package(OpenCV REQUIRED)
include_directories( ${OPENCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

find_package(OpenGL REQUIRED)
include_directories( ${OpenGL_INCLUDE_DIRS} )
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)

ADD_EXECUTABLE( test opengl_cv.cpp)
TARGET_LINK_LIBRARIES( test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

