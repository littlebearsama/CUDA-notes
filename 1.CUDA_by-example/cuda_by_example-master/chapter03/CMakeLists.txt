CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(enum_gpu enum_gpu.cu)
SET_PROPERTY(TARGET enum_gpu  PROPERTY FOLDER chapter03)

CUDA_ADD_EXECUTABLE(hello_world hello_world.cu)
SET_PROPERTY(TARGET hello_world  PROPERTY FOLDER chapter03)

CUDA_ADD_EXECUTABLE(set_gpu set_gpu.cu)
SET_PROPERTY(TARGET set_gpu  PROPERTY FOLDER chapter03)

CUDA_ADD_EXECUTABLE(simple_device_call simple_device_call.cu)
SET_PROPERTY(TARGET simple_device_call  PROPERTY FOLDER chapter03)

CUDA_ADD_EXECUTABLE(simple_kernel_params simple_kernel_params.cu)
SET_PROPERTY(TARGET simple_kernel_params  PROPERTY FOLDER chapter03)

CUDA_ADD_EXECUTABLE(simple_kernel simple_kernel.cu)
SET_PROPERTY(TARGET simple_kernel  PROPERTY FOLDER chapter03)