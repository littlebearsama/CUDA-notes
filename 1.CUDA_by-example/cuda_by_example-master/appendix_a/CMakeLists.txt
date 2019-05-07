CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(hashtable_cpu hashtable_cpu.cu)
SET_PROPERTY(TARGET hashtable_cpu  PROPERTY FOLDER appendix_a)

CUDA_ADD_EXECUTABLE(hashtable_gpu hashtable_gpu.cu lock.h)
SET_PROPERTY(TARGET hashtable_gpu PROPERTY FOLDER appendix_a)

CUDA_ADD_EXECUTABLE(dot_0 dot.cu  lock.h)
SET_PROPERTY(TARGET dot_0 PROPERTY FOLDER appendix_a)