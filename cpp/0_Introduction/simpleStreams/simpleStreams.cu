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
 * back to host (CPU) memory.
 *
 * The array is processed one chunk at a time, so that arrays much larger
 * than the available working buffers are still handled in full.  In single
 * stream mode chunks are processed strictly in order.  In multi-stream mode
 * each chunk runs H2D copy -> kernel -> D2H copy in its own stream, so that
 * transfers of one chunk can overlap with computation on another chunk.
 * Buffers are reused for a stream only after the previous chunk has been
 * copied back and its result has been read/verified on the host.
 *
 * Command line arguments (all "--name=<unsigned decimal integer>"):
 *   --n=<N>       number of elements to process        (1 .. 2^30,
 *                                                      default 2^24)
 *   --chunk=<N>   maximum elements processed per chunk (1 .. 2^24,
 *                                                      default 2^20)
 *   --streams=<N> number of streams for multi mode      (1 .. 16,
 *                                                      default 4)
 *   --repeat=<N>  number of repetitions                 (1 .. 100,
 *                                                      default 1)
 *
 * Working memory is bounded by 8*chunk*streams bytes on the device and the
 * same bound for pinned host memory, regardless of n and repeat.
 */

// System includes
#include <fenv.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

#define BLOCK 256 // threads per block

// Command line limits / defaults
#define N_MAX        (1LL << 30) // 1073741824
#define CHUNK_MAX    (1LL << 24) //   16777216
#define STREAMS_MAX  16
#define REPEAT_MAX   100

#define N_DEFAULT    (1LL << 24) //  16777216
#define CHUNK_DEFAULT (1LL << 20) //  1048576
#define STREAMS_DEFAULT 4
#define REPEAT_DEFAULT 1

// Device working memory may not exceed 512 MiB total:
// 2 float buffers * 4 bytes * chunk elements * streams
#define DEVICE_BUDGET_BYTES (512ULL << 20)

// Kernel: computes element-wise square of the input array
__global__ void square_kernel(const float* in, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}

// Report a failed CUDA call.  Returns false so callers can bail out; the
// caller is responsible for draining in-flight work and releasing resources.
static bool cuda_ok(cudaError_t status, const char* what)
{
    if (status == cudaSuccess)
        return true;
    fprintf(stderr, "CUDA error while %s: %s (%s)\n",
            what, cudaGetErrorString(status), cudaGetErrorName(status));
    return false;
}

// Parse one "--name=value" argument.  value must be a non-empty string of
// decimal digits with no sign, whitespace, or other characters.
static bool parse_decimal(const char* text, long long* value)
{
    if (text[0] == '\0')
        return false;
    unsigned long long v = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9')
            return false;
        unsigned long long d = (unsigned long long)(*p - '0');
        if (v > ((unsigned long long)-1 - d) / 10ULL)
            return false; // overflow
        v = v * 10ULL + d;
    }
    if (v > (unsigned long long)LLONG_MAX)
        return false;
    *value = (long long)v;
    return true;
}

struct Options {
    long long n;
    long long chunk;
    long long streams;
    long long repeat;
};

static bool parse_args(int argc, char** argv, Options* opt, const char** err)
{
    opt->n       = N_DEFAULT;
    opt->chunk   = CHUNK_DEFAULT;
    opt->streams = STREAMS_DEFAULT;
    opt->repeat  = REPEAT_DEFAULT;

    for (int a = 1; a < argc; ++a) {
        const char* arg = argv[a];
        const char* name = 0;
        const char* value = 0;

        if (strncmp(arg, "--n=", 4) == 0) {
            name = "n";       value = arg + 4;
        } else if (strncmp(arg, "--chunk=", 8) == 0) {
            name = "chunk";   value = arg + 8;
        } else if (strncmp(arg, "--streams=", 10) == 0) {
            name = "streams"; value = arg + 10;
        } else if (strncmp(arg, "--repeat=", 9) == 0) {
            name = "repeat";  value = arg + 9;
        } else {
            *err = "unrecognized or malformed argument (expected "
                   "--n/--chunk/--streams/--repeat=<decimal integer>)";
            return false;
        }

        long long v;
        if (!parse_decimal(value, &v)) {
            *err = "value must be a positive decimal integer";
            fprintf(stderr, "Invalid argument '--%s=%s': %s.\n",
                    name, value, *err);
            return false;
        }

        long long lo, hi;
        if (strcmp(name, "n") == 0)       { lo = 1; hi = N_MAX;       opt->n = v; }
        else if (strcmp(name, "chunk") == 0) { lo = 1; hi = CHUNK_MAX; opt->chunk = v; }
        else if (strcmp(name, "streams") == 0) { lo = 1; hi = STREAMS_MAX; opt->streams = v; }
        else                                 { lo = 1; hi = REPEAT_MAX; opt->repeat = v; }

        if (v < lo || v > hi) {
            *err = "value out of range";
            fprintf(stderr,
                    "Invalid argument '--%s=%lld': must be between %lld and %lld.\n",
                    name, v, lo, hi);
            return false;
        }
    }

    unsigned long long budget =
        8ULL * (unsigned long long)opt->chunk * (unsigned long long)opt->streams;
    if (budget > DEVICE_BUDGET_BYTES) {
        *err = "8*chunk*streams exceeds 512 MiB";
        fprintf(stderr,
                "Invalid arguments: 8*chunk*streams = %llu bytes exceeds the "
                "512 MiB (%u byte) device buffer budget.\n",
                budget, (unsigned)DEVICE_BUDGET_BYTES);
        return false;
    }
    return true;
}

// Reusable per-stream resources.  All working memory scales with chunk and
// streams only, never with n or repeat.
struct Context {
    int          numStreams;
    long long    chunk;
    cudaStream_t streams[STREAMS_MAX];
    cudaEvent_t  startEv;                 // first H2D of a round
    cudaEvent_t  endEv[STREAMS_MAX];      // last D2H per stream in a round
    float*       d_in[STREAMS_MAX];
    float*       d_out[STREAMS_MAX];
    float*       h_in[STREAMS_MAX];       // pinned, chunk floats each
    float*       h_out[STREAMS_MAX];      // pinned, chunk floats each
};

static bool setup_context(const Options& opt, Context* ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->numStreams = (int)opt.streams;
    ctx->chunk      = opt.chunk;

    size_t chunkBytes = (size_t)opt.chunk * sizeof(float);

    for (int s = 0; s < ctx->numStreams; ++s) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&ctx->streams[s],
                                               cudaStreamNonBlocking),
                     "creating stream"))
            return false;
        if (!cuda_ok(cudaEventCreate(&ctx->endEv[s]), "creating event"))
            return false;
        if (!cuda_ok(cudaMalloc(&ctx->d_in[s], chunkBytes),
                     "allocating device input buffer"))
            return false;
        if (!cuda_ok(cudaMalloc(&ctx->d_out[s], chunkBytes),
                     "allocating device output buffer"))
            return false;
        if (!cuda_ok(cudaMallocHost(&ctx->h_in[s], chunkBytes),
                     "allocating pinned host input buffer"))
            return false;
        if (!cuda_ok(cudaMallocHost(&ctx->h_out[s], chunkBytes),
                     "allocating pinned host output buffer"))
            return false;
    }
    if (!cuda_ok(cudaEventCreate(&ctx->startEv), "creating event"))
        return false;
    return true;
}

static void teardown_context(Context* ctx)
{
    // Make sure no work is still in flight before freeing anything.
    for (int s = 0; s < ctx->numStreams; ++s) {
        if (ctx->streams[s])
            (void)cudaStreamSynchronize(ctx->streams[s]);
    }
    for (int s = 0; s < ctx->numStreams; ++s) {
        if (ctx->d_in[s])    (void)cudaFree(ctx->d_in[s]);
        if (ctx->d_out[s])   (void)cudaFree(ctx->d_out[s]);
        if (ctx->h_in[s])    (void)cudaFreeHost(ctx->h_in[s]);
        if (ctx->h_out[s])   (void)cudaFreeHost(ctx->h_out[s]);
        if (ctx->endEv[s])   (void)cudaEventDestroy(ctx->endEv[s]);
        if (ctx->streams[s]) (void)cudaStreamDestroy(ctx->streams[s]);
    }
    if (ctx->startEv) (void)cudaEventDestroy(ctx->startEv);
}

// Fill the chunk starting at global element index base with
// float(globalIndex + round), and verify the returned squares against the
// CPU reference: square the float input and round back to float.
static void fill_input(float* buf, long long base, int count, long long round)
{
    for (int j = 0; j < count; ++j)
        buf[j] = (float)(base + (long long)j + round);
}

static bool verify_chunk(const float* got, long long base, int count,
                         long long round, const char* mode)
{
    for (int j = 0; j < count; ++j) {
        float x        = (float)(base + (long long)j + round);
        float expected = x * x; // rounded to float (round-to-nearest-even)
        uint32_t ebits, abits;
        memcpy(&ebits, &expected, sizeof(uint32_t));
        memcpy(&abits, &got[j],  sizeof(uint32_t));
        if (ebits != abits) {
            fprintf(stderr,
                    "Mismatch in round %lld, %s at global index %lld: "
                    "expected %.9g, got %.9g\n",
                    round, mode, (long long)(base + j),
                    (double)expected, (double)got[j]);
            return false;
        }
    }
    return true;
}

enum RunStatus { RUN_OK, RUN_CUDA_ERROR, RUN_MISMATCH };

// Number of elements in chunk c.
static inline int chunk_count(long long n, long long chunk, long long c)
{
    long long base = c * chunk;
    long long cnt  = n - base;
    if (cnt > chunk)
        cnt = chunk;
    return (int)cnt;
}

// Enqueue H2D -> kernel -> D2H for one chunk on stream slot s.
static bool submit_chunk(Context* ctx, int s, long long c, long long n,
                         long long round, bool recordEnd)
{
    int cnt = chunk_count(n, ctx->chunk, c);
    long long base = c * ctx->chunk;
    size_t bytes = (size_t)cnt * sizeof(float);
    int grid = (cnt + BLOCK - 1) / BLOCK;

    fill_input(ctx->h_in[s], base, cnt, round);

    if (!cuda_ok(cudaMemcpyAsync(ctx->d_in[s], ctx->h_in[s], bytes,
                                 cudaMemcpyHostToDevice, ctx->streams[s]),
                 "submitting H2D copy"))
        return false;
    square_kernel<<<grid, BLOCK, 0, ctx->streams[s]>>>(
        ctx->d_in[s], ctx->d_out[s], cnt);
    if (!cuda_ok(cudaPeekAtLastError(), "launching square kernel"))
        return false;
    if (!cuda_ok(cudaMemcpyAsync(ctx->h_out[s], ctx->d_out[s], bytes,
                                 cudaMemcpyDeviceToHost, ctx->streams[s]),
                 "submitting D2H copy"))
        return false;
    // Marks this stream's final D2H of the round.
    if (recordEnd &&
        !cuda_ok(cudaEventRecord(ctx->endEv[s], ctx->streams[s]),
                 "recording end event"))
        return false;
    return true;
}

// Wait for all in-flight streams, ignoring further errors (used on failure).
static void drain_all(Context* ctx)
{
    for (int s = 0; s < ctx->numStreams; ++s)
        (void)cudaStreamSynchronize(ctx->streams[s]);
}

// Single stream: chunks are processed strictly one after another, and each
// chunk is read back before its buffers are reused.
static RunStatus run_single(Context* ctx, long long n, long long round,
                            float* msOut)
{
    long long numChunks = (n + ctx->chunk - 1) / ctx->chunk;

    if (!cuda_ok(cudaEventRecord(ctx->startEv, ctx->streams[0]),
                 "recording start event"))
        return RUN_CUDA_ERROR;

    for (long long c = 0; c < numChunks; ++c) {
        bool last = (c == numChunks - 1);
        if (!submit_chunk(ctx, 0, c, n, round, last))
            return RUN_CUDA_ERROR;
        if (!cuda_ok(cudaStreamSynchronize(ctx->streams[0]),
                     "waiting for chunk")) {
            drain_all(ctx);
            return RUN_CUDA_ERROR;
        }
        int cnt = chunk_count(n, ctx->chunk, c);
        if (!verify_chunk(ctx->h_out[0], c * ctx->chunk, cnt, round,
                          "Single stream"))
            return RUN_MISMATCH;
    }

    if (!cuda_ok(cudaEventSynchronize(ctx->endEv[0]),
                 "synchronizing end event"))
        return RUN_CUDA_ERROR;
    if (!cuda_ok(cudaEventElapsedTime(msOut, ctx->startEv, ctx->endEv[0]),
                 "measuring elapsed time"))
        return RUN_CUDA_ERROR;
    return RUN_OK;
}

// Multi stream: a sliding window of at most numStreams chunks in flight.
// Chunk c is permanently assigned to slot c % numStreams, so each slot runs
// chunks s, s+S, s+2S, ... .  A slot's next chunk is submitted only after
// its previous chunk has been copied back and verified, so its buffers are
// never reused early; different slots still overlap freely.
static RunStatus run_multi(Context* ctx, long long n, long long round,
                           float* msOut)
{
    int S = ctx->numStreams;
    long long numChunks = (n + ctx->chunk - 1) / ctx->chunk;
    int usedStreams = (numChunks < S) ? (int)numChunks : S;

    long long slotChunk[STREAMS_MAX]; // chunk currently occupying slot s
    for (int s = 0; s < S; ++s)
        slotChunk[s] = -1;

    if (!cuda_ok(cudaEventRecord(ctx->startEv, ctx->streams[0]),
                 "recording start event"))
        return RUN_CUDA_ERROR;

    // Prime the pipeline with one chunk per used stream.
    for (int s = 0; s < usedStreams; ++s) {
        bool lastForSlot = ((long long)s + S >= numChunks);
        if (!submit_chunk(ctx, s, s, n, round, lastForSlot)) {
            drain_all(ctx);
            return RUN_CUDA_ERROR;
        }
        slotChunk[s] = s;
    }

    long long completed = 0;

    while (completed < numChunks) {
        bool progressed = false;
        for (int s = 0; s < usedStreams && completed < numChunks; ++s) {
            if (slotChunk[s] < 0)
                continue;

            cudaError_t q = cudaStreamQuery(ctx->streams[s]);
            if (q == cudaErrorNotReady)
                continue;
            if (q != cudaSuccess) {
                fprintf(stderr, "CUDA error while waiting for chunk %lld: "
                        "%s (%s)\n",
                        slotChunk[s], cudaGetErrorString(q),
                        cudaGetErrorName(q));
                drain_all(ctx);
                return RUN_CUDA_ERROR;
            }

            // D2H finished: read and verify before reusing the buffers.
            long long c = slotChunk[s];
            int cnt = chunk_count(n, ctx->chunk, c);
            if (!verify_chunk(ctx->h_out[s], c * ctx->chunk, cnt, round,
                              "Multi-stream")) {
                drain_all(ctx);
                return RUN_MISMATCH;
            }
            slotChunk[s] = -1;
            ++completed;
            progressed = true;

            long long nextC = c + S;
            if (nextC < numChunks) {
                bool lastForSlot = (nextC + S >= numChunks);
                if (!submit_chunk(ctx, s, nextC, n, round, lastForSlot)) {
                    drain_all(ctx);
                    return RUN_CUDA_ERROR;
                }
                slotChunk[s] = nextC;
            }
        }
        if (!progressed)
            sched_yield(); // all in-flight chunks still running
    }

    // Each slot's last D2H is marked by endEv[s]; the round ends at the
    // latest of them, so take the maximum elapsed time across all streams.
    float maxMs = 0.0f;
    for (int s = 0; s < usedStreams; ++s) {
        if (!cuda_ok(cudaEventSynchronize(ctx->endEv[s]),
                     "synchronizing end event"))
            return RUN_CUDA_ERROR;
        float ms;
        if (!cuda_ok(cudaEventElapsedTime(&ms, ctx->startEv, ctx->endEv[s]),
                     "measuring elapsed time"))
            return RUN_CUDA_ERROR;
        if (ms > maxMs)
            maxMs = ms;
    }
    *msOut = maxMs;
    return RUN_OK;
}

int main(int argc, char** argv)
{
    printf("[ CUDA Sample: Streams ]\n\n");

    Options opt;
    const char* err = 0;
    if (!parse_args(argc, argv, &opt, &err)) {
        if (err)
            fprintf(stderr, "Usage: simpleStreams [--n=<1..%lld>] "
                    "[--chunk=<1..%lld>] [--streams=<1..%d>] "
                    "[--repeat=<1..%d>]\n",
                    (long long)N_MAX, (long long)CHUNK_MAX, STREAMS_MAX,
                    REPEAT_MAX);
        return 1;
    }

    // CPU reference squares use the default round-to-nearest-even mode.
    fesetround(FE_TONEAREST);

    int deviceCount = 0;
    cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
    if (cerr != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No usable CUDA GPU available%s%s.\n",
                cerr != cudaSuccess ? ": " : "",
                cerr != cudaSuccess ? cudaGetErrorString(cerr) : "");
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(0), "selecting device 0"))
        return 1;

    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major,   cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor,   cudaDevAttrComputeCapabilityMinor, 0);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    printf("GPU Device 0: with compute capability %d.%d and Number of SMs %d\n",
           major, minor, smCount);
    printf("n=%lld chunk=%lld streams=%lld repeat=%lld\n\n",
           opt.n, opt.chunk, opt.streams, opt.repeat);

    Context ctx;
    if (!setup_context(opt, &ctx)) {
        teardown_context(&ctx);
        fprintf(stderr, "Failed to initialize CUDA resources.\n");
        return 1;
    }

    int exitCode = 0;

    for (long long r = 0; r < opt.repeat; ++r) {
        printf("Round %lld\n", r);

        float msSingle = 0.0f, msMulti = 0.0f;

        RunStatus s1 = run_single(&ctx, opt.n, r, &msSingle);
        if (s1 != RUN_OK) {
            if (s1 == RUN_CUDA_ERROR)
                fprintf(stderr, "Single stream failed in round %lld.\n", r);
            exitCode = 1;
            break;
        }
        printf("  Single stream = %lld elements, PASS, %.3f ms\n",
               opt.n, msSingle);

        RunStatus s2 = run_multi(&ctx, opt.n, r, &msMulti);
        if (s2 != RUN_OK) {
            if (s2 == RUN_CUDA_ERROR)
                fprintf(stderr, "Multi-stream failed in round %lld.\n", r);
            exitCode = 1;
            break;
        }
        printf("  Multi-stream  = %lld elements, PASS, %.3f ms\n",
               opt.n, msMulti);

        printf("  Speedup       = %.2fx\n",
               msMulti > 0.0f ? msSingle / msMulti : 0.0f);
        printf("\n");
    }

    teardown_context(&ctx);
    return exitCode;
}
