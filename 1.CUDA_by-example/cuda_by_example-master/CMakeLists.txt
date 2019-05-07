CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(cuda_by_example)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#outpath
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")  

#Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)


#CUDA
find_package(CUDA)
if (CUDA_FOUND)
	add_subdirectory(appendix_a)
	add_subdirectory(chapter03)
	add_subdirectory(chapter04)
	add_subdirectory(chapter05)
	add_subdirectory(chapter06)
	add_subdirectory(chapter07)
	add_subdirectory(chapter08)
	add_subdirectory(chapter09)
	add_subdirectory(chapter10)
else()
	message("CUDA not found!")
endif()
