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
 *
 * A is an M x K matrix, B is a K x N matrix and C is the row-major M x N
 * single-precision result. M, N and K may be any positive integers; the
 * kernel tolerates matrices whose extents are not multiples of the tile
 * size, including extents smaller than one tile.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <limits.h>
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

// Total device memory (A + B + C) a single process may allocate.
static const unsigned long long kMaxDeviceBufferBytes = 512ull * 1024 * 1024;

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * A is M x K, B is K x N, C is M x N (all row-major).
 *
 * Tiles that hang over the matrix edges are zero-filled, so arbitrary
 * M, N and K (including tails in the reduction dimension) are handled
 * without out-of-bounds accesses. The tile grid is linearized into
 * blockIdx.x so that very tall or very wide matrices cannot exceed the
 * 65535 limit of gridDim.y. Products of float inputs are accumulated in
 * double so that cancellation-heavy inputs still meet the tight
 * verification tolerance; the stored result is single precision.
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, const float *A, const float *B, int M, int N, int K, int tilesX)
{
    // Linearized tile indices: tileRow indexes M, tileCol indexes N.
    int tileRow = blockIdx.x / tilesX;
    int tileCol = blockIdx.x % tilesX;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row of C/A and column of C/B handled by this thread.
    int row = tileRow * BLOCK_SIZE + ty;
    int col = tileCol * BLOCK_SIZE + tx;

    // Csub accumulates the element of the block sub-matrix computed
    // by this thread, in double precision.
    double Csub = 0.0;

    // Loop over all the sub-matrices of A and B required to compute
    // the block sub-matrix, including a possibly partial tail tile.
    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory;
        // each thread loads one element of each matrix. Elements
        // outside the matrix are replaced with zero so that edge
        // tiles contribute nothing to the result.
        int aCol = k0 + tx;
        int bRow = k0 + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;

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
        C[static_cast<size_t>(row) * N + col] = static_cast<float>(Csub);
    }
}

// Input data modes.
enum DataMode { DATA_CONSTANT, DATA_MIXED };

void ConstantInit(float *data, size_t size, float val)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// Deterministic mixed-sign inputs, generated in float arithmetic from
// zero-based row/column indices so that positive and negative terms
// cancel: A[r,k] = s[k%4] + ((r+k)%7)*0.125 with
// s = [10000, 1, -10000, 0.001], B[k,c] = 1 + ((k+c)%3)*0.0625.
void MixedInitA(float *data, int M, int K)
{
    const float s[4] = {10000.0f, 1.0f, -10000.0f, 0.001f};

    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            data[static_cast<size_t>(r) * K + k] = s[k % 4] + static_cast<float>((r + k) % 7) * 0.125f;
        }
    }
}

void MixedInitB(float *data, int K, int N)
{
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < N; ++c) {
            data[static_cast<size_t>(k) * N + c] = 1.0f + static_cast<float>((k + c) % 3) * 0.0625f;
        }
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA.
 * A is M x K, B is K x N, C is M x N.
 */
int MatrixMultiply(int argc, char **argv, int block_size, int M, int N, int K, DataMode mode)
{
    // Allocate host memory for matrices A and B
    size_t size_A     = static_cast<size_t>(M) * K;
    size_t mem_size_A = sizeof(float) * size_A;
    float  *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A, 0));
    size_t size_B     = static_cast<size_t>(K) * N;
    size_t mem_size_B = sizeof(float) * size_B;
    float  *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B, 0));
    cudaStream_t stream;

    // Initialize host memory
    if (mode == DATA_MIXED) {
        MixedInitA(h_A, M, K);
        MixedInitB(h_B, K, N);
    }
    else {
        ConstantInit(h_A, size_A, 1.0f);
        ConstantInit(h_B, size_B, 0.01f);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    size_t mem_size_C = static_cast<size_t>(M) * N * sizeof(float);
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C, 0));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

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

    // Setup execution parameters. The tile grid is linearized along
    // gridDim.x so neither matrix extent can hit the 65535-block limit
    // of gridDim.y.
    dim3 threads(block_size, block_size);
    int  tilesX = (N + block_size - 1) / block_size;
    int  tilesY = (M + block_size - 1) / block_size;
    dim3 grid(static_cast<unsigned int>(tilesX) * static_cast<unsigned int>(tilesY));

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, tilesX);
    }
    else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, tilesX);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel. Scale the iteration count down for very large
    // problems so the timing loop stays bounded; timing is informational
    // only and never decides success.
    double flopsPerMatrixMul = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    int    nIter             = static_cast<int>(6.0e10 / flopsPerMatrixMul);

    if (nIter < 1) {
        nIter = 1;
    }
    else if (nIter > 300) {
        nIter = 300;
    }

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, tilesX);
        }
        else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, M, N, K, tilesX);
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
    double gigaFlops        = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul,
           threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Compute the double-precision CPU reference element by element from
    // the actual inputs, for both data modes.
    printf("Computing double-precision CPU reference...\n");
    std::vector<double> ref_C(static_cast<size_t>(M) * N, 0.0);

    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            double        a      = static_cast<double>(h_A[static_cast<size_t>(r) * K + k]);
            const float  *b_row  = h_B + static_cast<size_t>(k) * N;
            double       *ref_row = ref_C.data() + static_cast<size_t>(r) * N;

            for (int c = 0; c < N; ++c) {
                ref_row[c] += a * static_cast<double>(b_row[c]);
            }
        }
    }

    printf("Checking computed result for correctness: \n");
    bool correct = true;

    // Every element must satisfy |gpu - ref| <= 1e-5 + 1e-5*|ref|.
    // NaN or infinite GPU values always fail.
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        float  gpu_value = h_C[i];
        double ref_value = ref_C[i];

        if (!isfinite(gpu_value)) {
            printf("Error! Matrix[%d,%d]=%.8f is not finite, ref=%.8f\n",
                   static_cast<int>(i / N),
                   static_cast<int>(i % N),
                   gpu_value,
                   ref_value);
            correct = false;
            break;
        }

        double abs_err = fabs(static_cast<double>(gpu_value) - ref_value);
        double eps     = 1.e-5 + 1.e-5 * fabs(ref_value);

        if (abs_err > eps) {
            printf("Error! Matrix[%d,%d]=%.8f, ref=%.8f error term is > %E\n",
                   static_cast<int>(i / N),
                   static_cast<int>(i % N),
                   gpu_value,
                   ref_value,
                   eps);
            correct = false;
            break;
        }
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

/**
 * Parse a strictly positive integer option of the form -<name>=<digits>.
 * Sets *value when the option is present (leaving it untouched otherwise).
 * Any malformed, missing, non-numeric, non-positive or out-of-range value
 * is reported and aborts the program before any matrix is allocated.
 */
void ParsePositiveIntOption(int argc, char **argv, const char *name, int *value)
{
    size_t name_len = strlen(name);

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        while (*arg == '-') {
            ++arg;
        }

        if (strncmp(arg, name, name_len) != 0) {
            continue;
        }

        const char *rest = arg + name_len;

        if (*rest == '\0') {
            fprintf(stderr, "Error: option -%s requires a value (-%s=N).\n", name, name);
            exit(EXIT_FAILURE);
        }

        if (*rest != '=') {
            // A different option that merely shares a prefix.
            continue;
        }

        const char *value_str = rest + 1;

        if (*value_str == '\0') {
            fprintf(stderr, "Error: option -%s has an empty value.\n", name);
            exit(EXIT_FAILURE);
        }

        errno     = 0;
        char *end = NULL;
        long  parsed = strtol(value_str, &end, 10);

        if (errno != 0 || end == value_str || *end != '\0') {
            fprintf(stderr, "Error: option -%s has a non-numeric or out-of-range value '%s'.\n", name, value_str);
            exit(EXIT_FAILURE);
        }

        if (parsed <= 0) {
            fprintf(stderr, "Error: option -%s must be a positive integer, got '%s'.\n", name, value_str);
            exit(EXIT_FAILURE);
        }

        if (parsed > INT_MAX) {
            fprintf(stderr, "Error: option -%s=%s exceeds the representable range.\n", name, value_str);
            exit(EXIT_FAILURE);
        }

        *value = static_cast<int>(parsed);
        return;
    }
}

/**
 * Parse -data=constant|mixed. Defaults to constant; an unknown or
 * valueless mode aborts the program before any matrix is allocated.
 */
DataMode ParseDataMode(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        while (*arg == '-') {
            ++arg;
        }

        if (strncmp(arg, "data", 4) != 0) {
            continue;
        }

        const char *rest = arg + 4;

        if (*rest == '\0') {
            fprintf(stderr, "Error: option -data requires a value (-data=constant or -data=mixed).\n");
            exit(EXIT_FAILURE);
        }

        if (*rest != '=') {
            continue;
        }

        const char *value_str = rest + 1;

        if (strcmp(value_str, "constant") == 0) {
            return DATA_CONSTANT;
        }

        if (strcmp(value_str, "mixed") == 0) {
            return DATA_MIXED;
        }

        fprintf(stderr,
                "Error: unknown data mode '%s' (expected -data=constant or -data=mixed).\n",
                value_str);
        exit(EXIT_FAILURE);
    }

    return DATA_CONSTANT;
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
        printf("      -data=constant|mixed (input data pattern, default constant)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices"
               " must be equal (WidthA == HeightB).\n");
        printf("  Note: All dimensions may be any positive integers; they need"
               " not be multiples of the 32x32 thread-block size.\n");
        printf("  Note: Total device buffer size is limited to 512 MiB.\n");

        exit(EXIT_SUCCESS);
    }

    // Validate every size parameter before touching the GPU or
    // allocating any matrix.
    int block_size = 32;

    int wA = 5 * 2 * block_size, hA = 5 * 2 * block_size;
    int wB = 5 * 4 * block_size, hB = 5 * 2 * block_size;

    ParsePositiveIntOption(argc, argv, "wA", &wA);
    ParsePositiveIntOption(argc, argv, "hA", &hA);
    ParsePositiveIntOption(argc, argv, "wB", &wB);
    ParsePositiveIntOption(argc, argv, "hB", &hB);
    DataMode mode = ParseDataMode(argc, argv);

    // A is M x K, B is K x N, C is M x N.
    int M = hA, K = wA, N = wB;

    if (wA != hB) {
        fprintf(stderr, "Error: outer matrix dimensions must be equal. (%d != %d)\n", wA, hB);
        exit(EXIT_FAILURE);
    }

    // Reject requests whose device buffers would overflow size
    // computations or exceed the 512 MiB per-process limit, before any
    // allocation takes place.
    unsigned long long bytes_A = static_cast<unsigned long long>(M) * K * sizeof(float);
    unsigned long long bytes_B = static_cast<unsigned long long>(K) * N * sizeof(float);
    unsigned long long bytes_C = static_cast<unsigned long long>(M) * N * sizeof(float);
    unsigned long long bytes_total = bytes_A + bytes_B + bytes_C;

    if (bytes_total > kMaxDeviceBufferBytes) {
        fprintf(stderr,
                "Error: requested matrices need %llu bytes of device memory, "
                "exceeding the %llu byte (512 MiB) per-process limit.\n",
                bytes_total,
                kMaxDeviceBufferBytes);
        exit(EXIT_FAILURE);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", wA, hA, wB, hB);
    printf("C = A * B with M=%d, K=%d, N=%d\n", M, K, N);
    printf("Data mode: %s\n", mode == DATA_MIXED ? "mixed" : "constant");

#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStart());
#endif
    int matrix_result = MatrixMultiply(argc, argv, block_size, M, N, K, mode);
#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStop());
#endif

    exit(matrix_result);
}
