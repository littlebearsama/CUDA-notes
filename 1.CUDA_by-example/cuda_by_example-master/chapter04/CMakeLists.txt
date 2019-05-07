CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(add_loop_cpu add_loop_cpu.cu)
SET_PROPERTY(TARGET add_loop_cpu  PROPERTY FOLDER chapter04)

CUDA_ADD_EXECUTABLE(add_loop_gpu add_loop_gpu.cu)
SET_PROPERTY(TARGET add_loop_gpu  PROPERTY FOLDER chapter04)

CUDA_ADD_EXECUTABLE(add_loop_long add_loop_long.cu)
SET_PROPERTY(TARGET add_loop_long  PROPERTY FOLDER chapter04)

CUDA_ADD_EXECUTABLE(julia_cpu julia_cpu.cu)
SET_PROPERTY(TARGET julia_cpu  PROPERTY FOLDER chapter04)
TARGET_LINK_LIBRARIES(julia_cpu ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(julia_gpu julia_gpu.cu)
SET_PROPERTY(TARGET julia_gpu  PROPERTY FOLDER chapter04)
TARGET_LINK_LIBRARIES(julia_gpu ${OpenCV_LIBS})