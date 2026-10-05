# CMake generated Testfile for 
# Source directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001
# Build directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("vectorAdd_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/build/vectorAdd")
set_tests_properties("vectorAdd_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;18;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;0;")
add_test("matrixMul_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/build/matrixMul" "-wA=128" "-hA=128" "-wB=128" "-hB=128")
set_tests_properties("matrixMul_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;19;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;0;")
add_test("reduction_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/build/reduction" "--n=65536")
set_tests_properties("reduction_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;20;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;0;")
add_test("simpleStreams_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/build/simpleStreams")
set_tests_properties("simpleStreams_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;21;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task001/CMakeLists.txt;0;")
