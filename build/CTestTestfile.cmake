# CMake generated Testfile for 
# Source directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002
# Build directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("vectorAdd_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/vectorAdd")
set_tests_properties("vectorAdd_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;18;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=128" "-hA=128" "-wB=128" "-hB=128")
set_tests_properties("matrixMul_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;19;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_rect_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=100" "-hA=37" "-wB=53" "-hB=100")
set_tests_properties("matrixMul_rect_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;21;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_mixed_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=100" "-hA=37" "-wB=53" "-hB=100" "-data=mixed")
set_tests_properties("matrixMul_mixed_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;23;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_tiny_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=1" "-hA=1" "-wB=1" "-hB=1")
set_tests_properties("matrixMul_tiny_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;25;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_zero" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=0" "-hA=64" "-wB=64" "-hB=64")
set_tests_properties("matrixMul_bad_zero" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;27;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_k" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-wA=64" "-hA=64" "-wB=64" "-hB=32")
set_tests_properties("matrixMul_bad_k" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;28;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_data" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/matrixMul" "-data=unknown")
set_tests_properties("matrixMul_bad_data" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;29;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("reduction_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/reduction" "--n=65536")
set_tests_properties("reduction_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;30;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("simpleStreams_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/simpleStreams" "--n=1000003" "--chunk=65536" "--streams=4" "--repeat=2")
set_tests_properties("simpleStreams_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;32;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("simpleStreams_small_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/simpleStreams" "--n=1000" "--chunk=1048576" "--streams=8")
set_tests_properties("simpleStreams_small_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;34;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("simpleStreams_bad_n" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/simpleStreams" "--n=0")
set_tests_properties("simpleStreams_bad_n" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;36;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
add_test("simpleStreams_bad_limit" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/build/simpleStreams" "--chunk=16777216" "--streams=8")
set_tests_properties("simpleStreams_bad_limit" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;37;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/b/task002/CMakeLists.txt;0;")
