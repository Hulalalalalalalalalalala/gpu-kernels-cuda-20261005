cmake_minimum_required(VERSION 3.20)
project(gpu_kernel_examples LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
include_directories(Common)
enable_testing()
add_executable(vectorAdd cpp/0_Introduction/vectorAdd/vectorAdd.cu)
add_executable(matrixMul cpp/0_Introduction/matrixMul/matrixMul.cu)
add_executable(simpleStreams cpp/0_Introduction/simpleStreams/simpleStreams.cu)
add_executable(reduction cpp/2_Concepts_and_Techniques/reduction/reduction.cpp cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)
add_executable(transpose cpp/6_Performance/transpose/transpose.cu)
foreach(t vectorAdd matrixMul simpleStreams reduction transpose)
  target_link_libraries(${t} PRIVATE CUDA::cudart_static)
  target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endforeach()
add_test(NAME vectorAdd_gpu COMMAND vectorAdd)
add_test(NAME matrixMul_gpu COMMAND matrixMul -wA=128 -hA=128 -wB=128 -hB=128)
# rectangular matrices whose extents are not multiples of the 32x32 block
add_test(NAME matrixMul_rect_gpu COMMAND matrixMul -wA=100 -hA=37 -wB=53 -hB=100)
# mixed-sign inputs with a tail in the reduction dimension
add_test(NAME matrixMul_mixed_gpu COMMAND matrixMul -wA=100 -hA=37 -wB=53 -hB=100 -data=mixed)
# single-element matrices, smaller than one block in every dimension
add_test(NAME matrixMul_tiny_gpu COMMAND matrixMul -wA=1 -hA=1 -wB=1 -hB=1)
# invalid parameters must fail before any GPU compute
add_test(NAME matrixMul_bad_zero COMMAND matrixMul -wA=0 -hA=64 -wB=64 -hB=64)
add_test(NAME matrixMul_bad_k COMMAND matrixMul -wA=64 -hA=64 -wB=64 -hB=32)
add_test(NAME matrixMul_bad_data COMMAND matrixMul -data=unknown)
add_test(NAME reduction_gpu COMMAND reduction --n=65536)
# tail chunk (16963 elements, not a multiple of 256) and two rounds
add_test(NAME simpleStreams_gpu COMMAND simpleStreams --n=1000003 --chunk=65536 --streams=4 --repeat=2)
# n smaller than chunk and fewer chunks than streams
add_test(NAME simpleStreams_small_gpu COMMAND simpleStreams --n=1000 --chunk=1048576 --streams=8)
# invalid parameter values must fail before any GPU compute
add_test(NAME simpleStreams_bad_n COMMAND simpleStreams --n=0)
add_test(NAME simpleStreams_bad_limit COMMAND simpleStreams --chunk=16777216 --streams=8)
set_tests_properties(simpleStreams_bad_n simpleStreams_bad_limit matrixMul_bad_zero matrixMul_bad_k matrixMul_bad_data PROPERTIES WILL_FAIL TRUE)
set_tests_properties(vectorAdd_gpu matrixMul_gpu matrixMul_rect_gpu matrixMul_mixed_gpu matrixMul_tiny_gpu reduction_gpu simpleStreams_gpu simpleStreams_small_gpu PROPERTIES RUN_SERIAL TRUE)
