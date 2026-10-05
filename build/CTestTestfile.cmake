# CMake generated Testfile for 
# Source directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001
# Build directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("vectorAdd_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/vectorAdd")
set_tests_properties("vectorAdd_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;18;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("matrixMul_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/matrixMul" "-wA=128" "-hA=128" "-wB=128" "-hB=128")
set_tests_properties("matrixMul_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;19;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("reduction_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/reduction" "--n=65536")
set_tests_properties("reduction_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;20;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("simpleStreams_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/simpleStreams" "--n=1000003" "--chunk=65536" "--streams=4" "--repeat=2")
set_tests_properties("simpleStreams_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;22;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("simpleStreams_small_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/simpleStreams" "--n=1000" "--chunk=1048576" "--streams=8")
set_tests_properties("simpleStreams_small_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;24;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("simpleStreams_bad_n" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/simpleStreams" "--n=0")
set_tests_properties("simpleStreams_bad_n" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;26;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
add_test("simpleStreams_bad_limit" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/build/simpleStreams" "--chunk=16777216" "--streams=8")
set_tests_properties("simpleStreams_bad_limit" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;27;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task001/CMakeLists.txt;0;")
