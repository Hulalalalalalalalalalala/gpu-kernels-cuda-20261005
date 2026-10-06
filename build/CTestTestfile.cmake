# CMake generated Testfile for 
# Source directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002
# Build directory: /home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("vectorAdd_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/vectorAdd")
set_tests_properties("vectorAdd_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;18;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=128" "-hA=128" "-wB=128" "-hB=128")
set_tests_properties("matrixMul_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;19;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_rect_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=130" "-hA=100" "-wB=70" "-hB=130")
set_tests_properties("matrixMul_rect_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;21;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_tiny_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=1" "-hA=1" "-wB=1" "-hB=1")
set_tests_properties("matrixMul_tiny_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;23;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_mixed_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=1000" "-hA=257" "-wB=129" "-hB=1000" "-data=mixed")
set_tests_properties("matrixMul_mixed_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;25;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_size" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=0" "-hA=10" "-wB=10" "-hB=10")
set_tests_properties("matrixMul_bad_size" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;27;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_k" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=64" "-hA=64" "-wB=64" "-hB=32")
set_tests_properties("matrixMul_bad_k" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;28;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_bad_data" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-data=bogus")
set_tests_properties("matrixMul_bad_data" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;29;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("matrixMul_over_limit" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/matrixMul" "-wA=7000" "-hA=7000" "-wB=7000" "-hB=7000")
set_tests_properties("matrixMul_over_limit" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;30;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("reduction_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/reduction" "--n=65536")
set_tests_properties("reduction_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;31;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("simpleStreams_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/simpleStreams" "--n=1000003" "--chunk=65536" "--streams=4" "--repeat=2")
set_tests_properties("simpleStreams_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;33;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("simpleStreams_small_gpu" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/simpleStreams" "--n=1000" "--chunk=1048576" "--streams=8")
set_tests_properties("simpleStreams_small_gpu" PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;35;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("simpleStreams_bad_n" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/simpleStreams" "--n=0")
set_tests_properties("simpleStreams_bad_n" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;37;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
add_test("simpleStreams_bad_limit" "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/build/simpleStreams" "--chunk=16777216" "--streams=8")
set_tests_properties("simpleStreams_bad_limit" PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;38;add_test;/home/sl_chen/projects/gpu-kernels-cuda-20261005/a/task002/CMakeLists.txt;0;")
