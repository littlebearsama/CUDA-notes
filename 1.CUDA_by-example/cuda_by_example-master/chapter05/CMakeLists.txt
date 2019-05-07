CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(add_loop_blocks add_loop_blocks.cu)
SET_PROPERTY(TARGET add_loop_blocks  PROPERTY FOLDER chapter05)

CUDA_ADD_EXECUTABLE(add_loop_long_blocks add_loop_long_blocks.cu)
SET_PROPERTY(TARGET add_loop_long_blocks  PROPERTY FOLDER chapter05)

CUDA_ADD_EXECUTABLE(dot_5 dot.cu)
SET_PROPERTY(TARGET dot_5 PROPERTY FOLDER chapter05)

CUDA_ADD_EXECUTABLE(ripple_5 ripple.cu)
SET_PROPERTY(TARGET ripple_5  PROPERTY FOLDER chapter05)
TARGET_LINK_LIBRARIES(ripple_5 ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(shared_bitmap shared_bitmap.cu)
SET_PROPERTY(TARGET shared_bitmap  PROPERTY FOLDER chapter05)
TARGET_LINK_LIBRARIES(shared_bitmap ${OpenCV_LIBS})