CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(hist_cpu hist_cpu.cu)
SET_PROPERTY(TARGET hist_cpu  PROPERTY FOLDER chapter09)

CUDA_ADD_EXECUTABLE(hist_gpu_gmem_atomics hist_gpu_gmem_atomics.cu)
SET_PROPERTY(TARGET hist_gpu_gmem_atomics  PROPERTY FOLDER chapter09)

CUDA_ADD_EXECUTABLE(hist_gpu_shmem_atomics hist_gpu_shmem_atomics.cu)
SET_PROPERTY(TARGET hist_gpu_shmem_atomics  PROPERTY FOLDER chapter09)

