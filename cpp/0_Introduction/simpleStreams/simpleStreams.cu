/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel computes an
 * element-wise square of a float array, after which the result is copied
 * back to host (CPU) memory.  The array is processed in chunks of at most
 * --chunk elements so that inputs much larger than the working buffers
 * can be handled: the single-stream path works through the chunks one at
 * a time, while the multi-stream path issues the H2D copy, kernel, and
 * D2H copy of each chunk on one of --streams independent streams so that
 * transfers and computation of different chunks overlap on the device.
 *
 * The input is generated on the fly: in round r (counting from 0) the
 * value at global index i is float(i + r), so no n-sized host array is
 * ever needed.  Both paths use the same input and each validates its own
 * output chunk-by-chunk against a CPU reference (the float input squared
 * and rounded to float).  Working memory is bounded: device buffers and
 * pinned host staging buffers each use exactly 8 * chunk * streams bytes
 * (one input and one output buffer of chunk floats per stream), device
 * buffers never exceed 512 MiB, and nothing grows with n or repeat.
 *
 * Command line (all values are decimal positive integers):
 *   --n=N        total elements per round   [1, 1073741824] (default 16777216)
 *   --chunk=C    max elements per chunk     [1, 16777216]   (default 1048576)
 *   --streams=S  number of CUDA streams     [1, 16]         (default 4)
 *   --repeat=R   number of rounds           [1, 100]        (default 1)
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 */

// System includes
#include <stdio.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

#define BLOCK 256 // threads per block

// Parameter defaults
#define DEFAULT_N 16777216LL
#define DEFAULT_CHUNK 1048576LL
#define DEFAULT_STREAMS 4
#define DEFAULT_REPEAT 1

// Parameter upper bounds (lower bound is 1 for all of them)
#define MAX_N 1073741824LL
#define MAX_CHUNK 16777216LL
#define MAX_STREAMS 16
#define MAX_REPEAT 100

// Total device buffer budget: 512 MiB
#define MAX_DEVICE_BYTES (512LL * 1024 * 1024)

typedef struct {
    long long n;   // total elements to process per round
    long long chunk;   // maximum elements per chunk
    int streams;       // number of CUDA streams / buffer slots
    int repeat;        // number of rounds
} Config;

typedef struct {
    float* h_in;                            // pinned staging input,  streams*chunk floats
    float* h_out;                           // pinned staging output, streams*chunk floats
    float* d_in;                            // device input,          streams*chunk floats
    float* d_out;                           // device output,         streams*chunk floats
    cudaStream_t streams[MAX_STREAMS];      // one stream per buffer slot
    int streams_created;
    cudaEvent_t start, stop;                // timing events
    int events_created;
} Resources;

typedef struct {
    int status;              // 0 = PASS, 1 = MISMATCH, -1 = CUDA error
    float ms;                // elapsed time, valid when status == 0
    long long completed;     // elements computed and verified
    long long mismatch_index;    // global index of first mismatch
    float mismatch_expected;     // CPU reference value at mismatch_index
    float mismatch_actual;       // GPU value at mismatch_index
} RunResult;

// Kernel: computes element-wise square of the input array
__global__ void square_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}

// Number of valid elements in the chunk starting at global index base
// (every chunk except possibly the last one holds exactly chunk elements).
static int chunk_count(const Config* cfg, long long base) {
    long long remaining = cfg->n - base;
    return (int)(remaining < cfg->chunk ? remaining : cfg->chunk);
}

// Fill one host staging slot with this chunk's input: float(global_index + round)
static void fill_chunk(float* dst, long long base, int cnt, int round) {
    for (int j = 0; j < cnt; ++j) {
        dst[j] = (float)(base + j + round);
    }
}

// CPU reference: square the float input and round the result to float.
// Returns the global index of the first mismatch, or -1 if the chunk matches.
static long long verify_chunk(const float* got, long long base, int cnt, int round,
                              float* expected, float* actual) {
    for (int j = 0; j < cnt; ++j) {
        float x = (float)(base + j + round);
        float ref = x * x;
        if (got[j] != ref) {
            *expected = ref;
            *actual = got[j];
            return base + j;
        }
    }
    return -1;
}

// Parse a non-empty string of decimal digits as a positive integer.
// The value saturates above MAX_N, which is >= every parameter's upper
// bound, so the range checks afterwards still reject it.
static int parse_positive_decimal(const char* s, long long* out) {
    if (s == NULL || *s == '\0') return -1;
    long long v = 0;
    for (const char* p = s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return -1;
        if (v <= MAX_N) v = v * 10 + (*p - '0');
    }
    *out = v;
    return 0;
}

static int parse_args(int argc, char** argv, Config* cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* name;
        const char* value;
        long long v;

        if (strncmp(arg, "--n=", 4) == 0) {
            name = "--n"; value = arg + 4;
        } else if (strncmp(arg, "--chunk=", 8) == 0) {
            name = "--chunk"; value = arg + 8;
        } else if (strncmp(arg, "--streams=", 10) == 0) {
            name = "--streams"; value = arg + 10;
        } else if (strncmp(arg, "--repeat=", 9) == 0) {
            name = "--repeat"; value = arg + 9;
        } else {
            fprintf(stderr,
                    "Error: unknown argument '%s' (expected --n=, --chunk=, "
                    "--streams= or --repeat= with a decimal integer value)\n", arg);
            return -1;
        }

        if (parse_positive_decimal(value, &v) != 0) {
            fprintf(stderr, "Error: %s requires a decimal positive integer value (got '%s')\n",
                    name, value);
            return -1;
        }

        if (strcmp(name, "--n") == 0) {
            if (v < 1 || v > MAX_N) {
                fprintf(stderr, "Error: --n=%s is out of range [1, %lld]\n", value, MAX_N);
                return -1;
            }
            cfg->n = v;
        } else if (strcmp(name, "--chunk") == 0) {
            if (v < 1 || v > MAX_CHUNK) {
                fprintf(stderr, "Error: --chunk=%s is out of range [1, %lld]\n", value, MAX_CHUNK);
                return -1;
            }
            cfg->chunk = v;
        } else if (strcmp(name, "--streams") == 0) {
            if (v < 1 || v > MAX_STREAMS) {
                fprintf(stderr, "Error: --streams=%s is out of range [1, %d]\n", value, MAX_STREAMS);
                return -1;
            }
            cfg->streams = (int)v;
        } else {
            if (v < 1 || v > MAX_REPEAT) {
                fprintf(stderr, "Error: --repeat=%s is out of range [1, %d]\n", value, MAX_REPEAT);
                return -1;
            }
            cfg->repeat = (int)v;
        }
    }
    return 0;
}

static void free_resources(Resources* res) {
    if (res->h_in != NULL) cudaFreeHost(res->h_in);
    if (res->h_out != NULL) cudaFreeHost(res->h_out);
    if (res->d_in != NULL) cudaFree(res->d_in);
    if (res->d_out != NULL) cudaFree(res->d_out);
    for (int s = 0; s < res->streams_created; ++s) cudaStreamDestroy(res->streams[s]);
    if (res->events_created > 0) cudaEventDestroy(res->start);
    if (res->events_created > 1) cudaEventDestroy(res->stop);
    memset(res, 0, sizeof(*res));
}

// Allocate the bounded working set: one input and one output buffer of
// chunk floats per stream, on both the host (pinned) and the device.
static int alloc_resources(const Config* cfg, Resources* res) {
    memset(res, 0, sizeof(*res));
    size_t bytes = (size_t)cfg->chunk * (size_t)cfg->streams * sizeof(float);
    cudaError_t err;

    if ((err = cudaMallocHost((void**)&res->h_in, bytes)) != cudaSuccess) goto fail;
    if ((err = cudaMallocHost((void**)&res->h_out, bytes)) != cudaSuccess) goto fail;
    if ((err = cudaMalloc((void**)&res->d_in, bytes)) != cudaSuccess) goto fail;
    if ((err = cudaMalloc((void**)&res->d_out, bytes)) != cudaSuccess) goto fail;
    for (int s = 0; s < cfg->streams; ++s) {
        if ((err = cudaStreamCreateWithFlags(&res->streams[s], cudaStreamNonBlocking)) != cudaSuccess) goto fail;
        res->streams_created++;
    }
    if ((err = cudaEventCreate(&res->start)) != cudaSuccess) goto fail;
    res->events_created++;
    if ((err = cudaEventCreate(&res->stop)) != cudaSuccess) goto fail;
    res->events_created++;
    return 0;

fail:
    fprintf(stderr, "Error: CUDA resource allocation failed: %s\n", cudaGetErrorString(err));
    free_resources(res);
    return -1;
}

// Run one round in a single stream: chunks are processed strictly one
// after another (fill, H2D copy, kernel, D2H copy, verify).
static RunResult run_single_stream(const Config* cfg, const Resources* res, int round) {
    RunResult rr;
    memset(&rr, 0, sizeof(rr));
    long long num_chunks = (cfg->n + cfg->chunk - 1) / cfg->chunk;
    cudaError_t err;

    // the timing window starts just before the first chunk's H2D copy
    if ((err = cudaEventRecord(res->start, res->streams[0])) != cudaSuccess) goto cuda_fail;

    for (long long c = 0; c < num_chunks; ++c) {
        long long base = c * cfg->chunk;
        int cnt = chunk_count(cfg, base);

        fill_chunk(res->h_in, base, cnt, round);

        if ((err = cudaMemcpyAsync(res->d_in, res->h_in, cnt * sizeof(float),
                                   cudaMemcpyHostToDevice, res->streams[0])) != cudaSuccess) goto cuda_fail;
        int grid = (cnt + BLOCK - 1) / BLOCK;
        square_kernel<<<grid, BLOCK, 0, res->streams[0]>>>(res->d_in, res->d_out, cnt);
        if ((err = cudaGetLastError()) != cudaSuccess) goto cuda_fail;
        if ((err = cudaMemcpyAsync(res->h_out, res->d_out, cnt * sizeof(float),
                                   cudaMemcpyDeviceToHost, res->streams[0])) != cudaSuccess) goto cuda_fail;
        if ((err = cudaStreamSynchronize(res->streams[0])) != cudaSuccess) goto cuda_fail;

        long long bad = verify_chunk(res->h_out, base, cnt, round,
                                     &rr.mismatch_expected, &rr.mismatch_actual);
        if (bad >= 0) {
            rr.status = 1;
            rr.mismatch_index = bad;
            return rr;
        }
        rr.completed += cnt;
    }

    // stop is recorded after the last chunk's D2H copy in the same stream
    if ((err = cudaEventRecord(res->stop, res->streams[0])) != cudaSuccess) goto cuda_fail;
    if ((err = cudaEventSynchronize(res->stop)) != cudaSuccess) goto cuda_fail;
    if ((err = cudaEventElapsedTime(&rr.ms, res->start, res->stop)) != cudaSuccess) goto cuda_fail;
    rr.status = 0;
    return rr;

cuda_fail:
    fprintf(stderr, "Error: CUDA call failed in single-stream round %d: %s\n",
            round, cudaGetErrorString(err));
    cudaStreamSynchronize(res->streams[0]); // best effort: drain submitted work
    rr.status = -1;
    return rr;
}

// Run one round split across cfg->streams streams: chunk c uses stream and
// buffer slot c % streams, so transfers and kernels of different chunks can
// overlap.  A slot is only reused after the previous chunk in it has been
// copied back and validated on the host.
static RunResult run_multi_stream(const Config* cfg, const Resources* res, int round) {
    RunResult rr;
    memset(&rr, 0, sizeof(rr));
    const int S = cfg->streams;
    long long num_chunks = (cfg->n + cfg->chunk - 1) / cfg->chunk;
    long long tail_start = (num_chunks > S) ? num_chunks - S : 0;
    cudaError_t err;

    // chunk 0 runs on stream 0, so this precedes the first H2D copy
    if ((err = cudaEventRecord(res->start, res->streams[0])) != cudaSuccess) goto cuda_fail;

    for (long long c = 0; c < num_chunks; ++c) {
        int s = (int)(c % S);
        float* h_in_slot = res->h_in + (size_t)s * cfg->chunk;
        float* h_out_slot = res->h_out + (size_t)s * cfg->chunk;
        float* d_in_slot = res->d_in + (size_t)s * cfg->chunk;
        float* d_out_slot = res->d_out + (size_t)s * cfg->chunk;

        if (c >= S) {
            // Slot reuse: wait until the previous chunk in this stream has
            // been copied back, then read and validate it before overwriting.
            if ((err = cudaStreamSynchronize(res->streams[s])) != cudaSuccess) goto cuda_fail;
            long long pbase = (c - S) * cfg->chunk;
            int pcnt = chunk_count(cfg, pbase);
            long long bad = verify_chunk(h_out_slot, pbase, pcnt, round,
                                         &rr.mismatch_expected, &rr.mismatch_actual);
            if (bad >= 0) {
                rr.status = 1;
                rr.mismatch_index = bad;
                return rr;
            }
            rr.completed += pcnt;
        }

        long long base = c * cfg->chunk;
        int cnt = chunk_count(cfg, base);

        fill_chunk(h_in_slot, base, cnt, round);

        // H2D: copy this chunk to the device (non-blocking on the host)
        if ((err = cudaMemcpyAsync(d_in_slot, h_in_slot, cnt * sizeof(float),
                                   cudaMemcpyHostToDevice, res->streams[s])) != cudaSuccess) goto cuda_fail;
        // kernel: starts only after the H2D copy in this stream completes
        int grid = (cnt + BLOCK - 1) / BLOCK;
        square_kernel<<<grid, BLOCK, 0, res->streams[s]>>>(d_in_slot, d_out_slot, cnt);
        if ((err = cudaGetLastError()) != cudaSuccess) goto cuda_fail;
        // D2H: copy the result back after the kernel in this stream finishes
        if ((err = cudaMemcpyAsync(h_out_slot, d_out_slot, cnt * sizeof(float),
                                   cudaMemcpyDeviceToHost, res->streams[s])) != cudaSuccess) goto cuda_fail;
    }

    // wait for every submitted chunk to finish before ending the round
    for (int s = 0; s < S; ++s) {
        if ((err = cudaStreamSynchronize(res->streams[s])) != cudaSuccess) goto cuda_fail;
    }

    // stop is recorded only after all streams have drained, so the timing
    // window covers the last chunk's D2H copy, not just one stream
    if ((err = cudaEventRecord(res->stop, res->streams[0])) != cudaSuccess) goto cuda_fail;
    if ((err = cudaEventSynchronize(res->stop)) != cudaSuccess) goto cuda_fail;
    if ((err = cudaEventElapsedTime(&rr.ms, res->start, res->stop)) != cudaSuccess) goto cuda_fail;

    // validate the chunks still sitting in the tail slots
    for (long long c = tail_start; c < num_chunks; ++c) {
        int s = (int)(c % S);
        long long base = c * cfg->chunk;
        int cnt = chunk_count(cfg, base);
        long long bad = verify_chunk(res->h_out + (size_t)s * cfg->chunk, base, cnt, round,
                                     &rr.mismatch_expected, &rr.mismatch_actual);
        if (bad >= 0) {
            rr.status = 1;
            rr.mismatch_index = bad;
            return rr;
        }
        rr.completed += cnt;
    }

    rr.status = 0;
    return rr;

cuda_fail:
    fprintf(stderr, "Error: CUDA call failed in multi-stream round %d: %s\n",
            round, cudaGetErrorString(err));
    // stop submitting new chunks; drain whatever was already submitted
    for (int s = 0; s < S; ++s) cudaStreamSynchronize(res->streams[s]);
    rr.status = -1;
    return rr;
}

int main(int argc, char** argv) {
    Config cfg = { DEFAULT_N, DEFAULT_CHUNK, DEFAULT_STREAMS, DEFAULT_REPEAT };
    if (parse_args(argc, argv, &cfg) != 0) return 1;

    // device buffers use 2 buffers * chunk floats * streams * 4 bytes
    long long device_bytes = 8LL * cfg.chunk * cfg.streams;
    if (device_bytes > MAX_DEVICE_BYTES) {
        fprintf(stderr,
                "Error: 8*chunk*streams = %lld bytes exceeds the 512 MiB device "
                "buffer limit; reduce --chunk or --streams\n", device_bytes);
        return 1;
    }

    printf("[ CUDA Sample: Streams ]\n\n");

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count < 1) {
        fprintf(stderr, "Error: no CUDA-capable GPU is available (%s)\n",
                err == cudaSuccess ? "device count is 0" : cudaGetErrorString(err));
        return 1;
    }

    // Query compute capability and number of SMs on device 0
    int cuda_device = 0;
    int major = 0, minor = 0, smCount = 0;
    if ((err = cudaSetDevice(cuda_device)) != cudaSuccess) goto no_gpu;
    if ((err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cuda_device)) != cudaSuccess) goto no_gpu;
    if ((err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cuda_device)) != cudaSuccess) goto no_gpu;
    if ((err = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, cuda_device)) != cudaSuccess) goto no_gpu;
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n",
           cuda_device, major, minor, smCount);
    printf("n=%lld chunk=%lld streams=%d repeat=%d\n\n",
           cfg.n, cfg.chunk, cfg.streams, cfg.repeat);

    {
        Resources res;
        if (alloc_resources(&cfg, &res) != 0) return 1;

        int exit_code = 0;
        for (int r = 0; r < cfg.repeat; ++r) {
            // time single-stream execution for reference
            RunResult single = run_single_stream(&cfg, &res, r);
            if (single.status == 1) {
                printf("Round %d Single stream: MISMATCH at global index %lld "
                       "(expected %.9g, got %.9g)\n",
                       r, single.mismatch_index, single.mismatch_expected, single.mismatch_actual);
                exit_code = 1;
                break;
            }
            if (single.status != 0) { exit_code = 1; break; }
            printf("Round %d Single stream: %lld elements PASS (%.3f ms)\n",
                   r, single.completed, single.ms);

            // time execution split across the configured streams
            RunResult multi = run_multi_stream(&cfg, &res, r);
            if (multi.status == 1) {
                printf("Round %d Multi-stream: MISMATCH at global index %lld "
                       "(expected %.9g, got %.9g)\n",
                       r, multi.mismatch_index, multi.mismatch_expected, multi.mismatch_actual);
                exit_code = 1;
                break;
            }
            if (multi.status != 0) { exit_code = 1; break; }
            printf("Round %d Multi-stream: %lld elements PASS (%.3f ms)\n",
                   r, multi.completed, multi.ms);

            // the speedup ratio is informational only
            printf("Round %d Speedup: %.2fx\n", r, single.ms / multi.ms);
        }

        // make sure all submitted work has finished before releasing resources
        cudaDeviceSynchronize();
        free_resources(&res);
        return exit_code;
    }

no_gpu:
    fprintf(stderr, "Error: CUDA device query failed: %s\n", cudaGetErrorString(err));
    return 1;
}
