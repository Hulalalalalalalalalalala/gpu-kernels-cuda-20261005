/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 *
 * The kernel supports arbitrary positive rectangular sizes: A is M x K,
 * B is K x N and C is M x N.  Tiles that hang over the last row, the last
 * column or the tail of the reduction dimension are guarded so every
 * element of C is computed exactly once and no out-of-bounds access occurs.
 */

// System includes
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
// The CUDA Safe runtime does not ship the profiler API; gate it on header availability.
#if defined(__has_include) && __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#define HAVE_CUDA_PROFILER_API 1
#endif

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Total device buffer space (d_A + d_B + d_C) a single process may request.
static const unsigned long long kMaxDeviceBufferBytes = 512ull * 1024 * 1024; // 512 MiB

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * A is M x K, B is K x N, C is M x N (all row-major).
 * blocksPerRow is the number of thread blocks along the N dimension; the
 * 1D grid is decomposed as (row block, column block) inside the kernel so
 * that arbitrarily large M and N stay within grid dimension limits.
 *
 * The reduction is accumulated in double precision so that inputs with
 * large cancelling terms still match the double-precision CPU reference
 * within the required tolerance; the output itself is single precision.
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, const float *A, const float *B, int M, int N, int K, int blocksPerRow)
{
    // Decompose the 1D block index into 2D tile coordinates.
    int bx = blockIdx.x % blocksPerRow;
    int by = blockIdx.x / blocksPerRow;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row/column of C computed by this thread.
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Declaration of the shared memory arrays used to
    // store the sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Csub is used to accumulate the element of the block
    // sub-matrix that is computed by the thread
    double Csub = 0.0;

    // Loop over all the sub-matrices of A and B required to compute
    // the block sub-matrix, including a partial tail tile when K is
    // not a multiple of BLOCK_SIZE.
    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
        // Load the tiles from device memory to shared memory; each
        // thread loads one element of each matrix.  Elements outside
        // the matrix bounds are zero-filled so partial edge tiles
        // contribute nothing to the result.
        int aCol = k0 + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[static_cast<long long>(row) * K + aCol] : 0.0f;

        int bRow = k0 + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[static_cast<long long>(bRow) * N + col] : 0.0f;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += static_cast<double>(As[ty][k]) * static_cast<double>(Bs[k][tx]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    if (row < M && col < N) {
        C[static_cast<long long>(row) * N + col] = static_cast<float>(Csub);
    }
}

void ConstantInit(float *data, long long size, float val)
{
    for (long long i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// Deterministic mixed-sign input pattern.  r, k, c are zero-based row and
// column indices; all arithmetic is performed in float.
void MixedInit(float *A, int M, int K, float *B, int N)
{
    const float s[4] = {10000.0f, 1.0f, -10000.0f, 0.001f};

    for (long long r = 0; r < M; ++r) {
        for (long long k = 0; k < K; ++k) {
            A[r * K + k] = s[k % 4] + static_cast<float>((r + k) % 7) * 0.125f;
        }
    }

    for (long long k = 0; k < K; ++k) {
        for (long long c = 0; c < N; ++c) {
            B[k * N + c] = 1.0f + static_cast<float>((k + c) % 3) * 0.0625f;
        }
    }
}

enum DataMode { DATA_CONSTANT, DATA_MIXED };

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB, DataMode mode)
{
    const int M = dimsA.y; // rows of A and C
    const int K = dimsA.x; // columns of A, rows of B
    const int N = dimsB.x; // columns of B and C

    // Allocate host memory for matrices A and B
    long long size_A     = static_cast<long long>(M) * K;
    size_t   mem_size_A = sizeof(float) * static_cast<size_t>(size_A);
    float       *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A, 0));
    long long size_B     = static_cast<long long>(K) * N;
    size_t   mem_size_B = sizeof(float) * static_cast<size_t>(size_B);
    float       *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B, 0));
    cudaStream_t stream;

    // Initialize host memory
    if (mode == DATA_MIXED) {
        MixedInit(h_A, M, K, h_B, N);
    }
    else {
        ConstantInit(h_A, size_A, 1.0f);
        ConstantInit(h_B, size_B, 0.01f);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    long long size_C     = static_cast<long long>(M) * N;
    size_t   mem_size_C = sizeof(float) * static_cast<size_t>(size_C);
    float       *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C, 0));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters: a 1D grid covers all tiles of C, so
    // dimensions that are not multiples of the block size still launch
    // exactly the tiles needed for the requested matrix.
    dim3 threads(block_size, block_size);
    unsigned int blocksPerRow = (static_cast<unsigned int>(N) + threads.x - 1) / threads.x;
    unsigned int blocksPerCol = (static_cast<unsigned int>(M) + threads.y - 1) / threads.y;
    dim3 grid(blocksPerRow * blocksPerCol);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, blocksPerRow);
    }
    else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, blocksPerRow);
    }
    checkCudaErrors(cudaGetLastError());

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel.  The iteration count is scaled down for very
    // large matrices so the total timed work stays bounded; timing is
    // informational only and never decides success.
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    int nIter = 300;

    if (flopsPerMatrixMul > 0.0) {
        double scaled = 3.0e11 / flopsPerMatrixMul;

        if (scaled < 300.0) {
            nIter = scaled < 1.0 ? 1 : static_cast<int>(scaled);
        }
    }

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, blocksPerRow);
        }
        else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, blocksPerRow);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float  msecPerMatrixMul = msecTotal / nIter;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul,
           threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // Every element is compared against a double-precision CPU reference
    // computed from the actual inputs.  The allowed absolute error is
    // 1e-5 + 1e-5 * |reference|; NaN or infinite GPU results always fail.
    // The reference is accumulated row by row over column tiles so that
    // accesses to B stay cache friendly, and rows are split across worker
    // threads; each worker records the first failing element it sees.
    long long fail_row = -1, fail_col = -1;
    float     fail_gpu = 0.0f;
    double    fail_ref = 0.0;

    {
        const long long kChunk = 1024;
        unsigned int  nWorkers = std::thread::hardware_concurrency();

        if (nWorkers == 0) {
            nWorkers = 1;
        }

        if (nWorkers > static_cast<unsigned int>(M)) {
            nWorkers = static_cast<unsigned int>(M);
        }

        std::vector<long long>   wRow(nWorkers, -1), wCol(nWorkers, -1);
        std::vector<float>       wGpu(nWorkers, 0.0f);
        std::vector<double>      wRef(nWorkers, 0.0);
        std::vector<std::thread> workers;

        for (unsigned int w = 0; w < nWorkers; ++w) {
            long long r0 = static_cast<long long>(M) * w / nWorkers;
            long long r1 = static_cast<long long>(M) * (w + 1) / nWorkers;

            workers.emplace_back([&, w, r0, r1]() {
                std::vector<double> ref(kChunk);

                for (long long r = r0; r < r1 && wRow[w] < 0; ++r) {
                    const float *aRow = h_A + r * static_cast<long long>(K);

                    for (long long c0 = 0; c0 < N && wRow[w] < 0; c0 += kChunk) {
                        long long cn = (N - c0 < kChunk) ? (N - c0) : kChunk;

                        for (long long i = 0; i < cn; ++i) {
                            ref[i] = 0.0;
                        }

                        for (long long k = 0; k < K; ++k) {
                            double       a    = static_cast<double>(aRow[k]);
                            const float *bRow = h_B + k * static_cast<long long>(N) + c0;

                            for (long long i = 0; i < cn; ++i) {
                                ref[i] += a * static_cast<double>(bRow[i]);
                            }
                        }

                        for (long long i = 0; i < cn; ++i) {
                            float  gpu = h_C[r * static_cast<long long>(N) + c0 + i];
                            double tol = 1e-5 + 1e-5 * fabs(ref[i]);

                            if (!isfinite(gpu) || fabs(static_cast<double>(gpu) - ref[i]) > tol) {
                                wRow[w] = r;
                                wCol[w] = c0 + i;
                                wGpu[w] = gpu;
                                wRef[w] = ref[i];
                                break;
                            }
                        }
                    }
                }
            });
        }

        for (auto &t : workers) {
            t.join();
        }

        // The first failing element in row-major order wins; each row is
        // owned by exactly one worker, so comparing rows is enough.
        for (unsigned int w = 0; w < nWorkers; ++w) {
            if (wRow[w] >= 0 && (fail_row < 0 || wRow[w] < fail_row)) {
                fail_row = wRow[w];
                fail_col = wCol[w];
                fail_gpu = wGpu[w];
                fail_ref = wRef[w];
            }
        }

        correct = (fail_row < 0);
    }

    if (!correct) {
        printf("\nError! Matrix[%lld,%lld]: GPU=%.8f, ref=%.8f, allowed abs error = %E\n",
               fail_row,
               fail_col,
               fail_gpu,
               fail_ref,
               1e-5 + 1e-5 * fabs(fail_ref));
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance "
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

// Returns the value text of "-name=value" / "--name=value", an empty string
// when the flag is present without a value, or NULL when it is absent.
static const char *FindCmdLineValue(int argc, char **argv, const char *name)
{
    size_t name_len = strlen(name);

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        while (*arg == '-') {
            ++arg;
        }

        if (strncmp(arg, name, name_len) == 0) {
            if (arg[name_len] == '=') {
                return arg + name_len + 1;
            }

            if (arg[name_len] == '\0') {
                return arg + name_len; // flag present but no value
            }
        }
    }

    return NULL;
}

// Strictly parses a positive matrix dimension.  Prints a specific message
// and returns false for missing, non-numeric, non-positive or
// out-of-representable-range values.
static bool ParseDimension(const char *text, const char *param, int *out)
{
    if (text == NULL || *text == '\0') {
        fprintf(stderr, "Error: parameter -%s requires a positive integer value.\n", param);
        return false;
    }

    const char *p    = text;
    bool        neg  = false;

    if (*p == '-') {
        neg = true;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }

    if (*p == '\0') {
        fprintf(stderr, "Error: parameter -%s has non-numeric value '%s'.\n", param, text);
        return false;
    }

    long long value = 0;

    for (; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            fprintf(stderr, "Error: parameter -%s has non-numeric value '%s'.\n", param, text);
            return false;
        }

        value = value * 10 + (*p - '0');

        if (value > 2147483647ll) {
            fprintf(stderr,
                    "Error: parameter -%s=%s exceeds the representable range (max 2147483647).\n",
                    param,
                    text);
            return false;
        }
    }

    if (neg) {
        value = -value;
    }

    if (value <= 0) {
        fprintf(stderr, "Error: parameter -%s must be a positive integer (got '%s').\n", param, text);
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("      -data=constant|mixed (input data pattern, default: constant)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices"
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    // Validate every requested dimension before touching any matrix.
    int parsed;

    if (FindCmdLineValue(argc, argv, "wA") != NULL) {
        if (!ParseDimension(FindCmdLineValue(argc, argv, "wA"), "wA", &parsed)) {
            exit(EXIT_FAILURE);
        }
        dimsA.x = parsed;
    }

    if (FindCmdLineValue(argc, argv, "hA") != NULL) {
        if (!ParseDimension(FindCmdLineValue(argc, argv, "hA"), "hA", &parsed)) {
            exit(EXIT_FAILURE);
        }
        dimsA.y = parsed;
    }

    if (FindCmdLineValue(argc, argv, "wB") != NULL) {
        if (!ParseDimension(FindCmdLineValue(argc, argv, "wB"), "wB", &parsed)) {
            exit(EXIT_FAILURE);
        }
        dimsB.x = parsed;
    }

    if (FindCmdLineValue(argc, argv, "hB") != NULL) {
        if (!ParseDimension(FindCmdLineValue(argc, argv, "hB"), "hB", &parsed)) {
            exit(EXIT_FAILURE);
        }
        dimsB.y = parsed;
    }

    // Input data pattern: constant (default) or mixed.
    DataMode    mode      = DATA_CONSTANT;
    const char *data_mode = FindCmdLineValue(argc, argv, "data");

    if (data_mode != NULL) {
        if (strcmp(data_mode, "constant") == 0) {
            mode = DATA_CONSTANT;
        }
        else if (strcmp(data_mode, "mixed") == 0) {
            mode = DATA_MIXED;
        }
        else {
            fprintf(stderr,
                    "Error: unknown or missing data mode '%s' (expected -data=constant or -data=mixed).\n",
                    data_mode);
            exit(EXIT_FAILURE);
        }
    }

    if (dimsA.x != dimsB.y) {
        fprintf(stderr,
                "Error: outer matrix dimensions must be equal (wA=%d != hB=%d).\n",
                dimsA.x,
                dimsB.y);
        exit(EXIT_FAILURE);
    }

    // Reject requests whose device buffers would exceed the per-process
    // limit.  Dimensions are positive ints, so the element products and
    // their sum always fit in an unsigned 64-bit integer without overflow.
    unsigned long long elems_A = static_cast<unsigned long long>(dimsA.x) * dimsA.y;
    unsigned long long elems_B = static_cast<unsigned long long>(dimsB.x) * dimsB.y;
    unsigned long long elems_C = static_cast<unsigned long long>(dimsB.x) * dimsA.y;
    unsigned long long total   = elems_A + elems_B + elems_C;

    if (total > kMaxDeviceBufferBytes / sizeof(float)) {
        fprintf(stderr,
                "Error: requested matrices need %.1f MiB of device buffer space, "
                "exceeding the %llu MiB per-process limit.\n",
                4.0 * static_cast<double>(total) / (1024.0 * 1024.0),
                kMaxDeviceBufferBytes / (1024ull * 1024ull));
        exit(EXIT_FAILURE);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    printf("MatrixA(%d,%d), MatrixB(%d,%d), Data=%s\n",
           dimsA.x,
           dimsA.y,
           dimsB.x,
           dimsB.y,
           mode == DATA_MIXED ? "mixed" : "constant");

#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStart());
#endif
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, mode);
#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStop());
#endif

    exit(matrix_result);
}
