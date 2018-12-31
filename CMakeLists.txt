cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuSNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
option(ENABLE_STATIC "Build static (.a) library" ON)

add_library(cuSNN SHARED
	        src/cusnn_kernels.cu
            src/cusnn_kernels.cuh
            src/cusnn.cu
            src/cusnn.cuh)

target_compile_features(cuSNN PUBLIC cxx_std_11)
set_target_properties(cuSNN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
install(TARGETS "cuSNN" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(ENABLE_STATIC)
    add_library(cuSNN-static STATIC 
	            src/cusnn_kernels.cu
                src/cusnn_kernels.cuh
                src/cusnn.cu
                src/cusnn.cuh)
    set_target_properties(cuSNN-static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(cuSNN-static PROPERTIES OUTPUT_NAME "cuSNN")
    install(TARGETS "cuSNN-static" ARCHIVE DESTINATION lib)
endif(ENABLE_STATIC)

install(FILES "src/cusnn.cuh" DESTINATION include)
install(FILES "src/cusnn_kernels.cuh" DESTINATION include)
