CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(basic_double_stream_correct basic_double_stream_correct.cu)
SET_PROPERTY(TARGET basic_double_stream_correct  PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(basic_double_stream basic_double_stream.cu)
SET_PROPERTY(TARGET basic_double_stream  PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(basic_single_stream basic_single_stream.cu)
SET_PROPERTY(TARGET basic_single_stream  PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(copy_timed copy_timed.cu)
SET_PROPERTY(TARGET copy_timed  PROPERTY FOLDER chapter10)