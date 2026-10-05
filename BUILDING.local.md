# Native build and baseline verification

Use native entry points below. Build tooling and test selection are distinct from product requirements. Target task-specific cases and nearby regressions, record selected/filtered counts and exit codes. Zero selected tests is not success.

GPU required: local RTX5080 Laptop, sm_120. CUDA13.2.2 installed under /home/sl_chen/.local/cuda-13.2.2; no Python wrapper or CPU fallback may stand in for actual GPU execution. This snapshot includes five native upstream examples and Common headers, with a focused CMake entry. Tests compare numeric GPU results with CPU references; performance timings are informational only. Limit per-process device buffers to 512MiB and run tests serially, no timing thresholds.

```sh
/home/sl_chen/.local/bin/cmake -S . -B build -DCMAKE_CUDA_COMPILER=/home/sl_chen/.local/cuda-13.2.2/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=120
/home/sl_chen/.local/bin/cmake --build build -j2
/home/sl_chen/.local/bin/ctest --test-dir build --output-on-failure
```
