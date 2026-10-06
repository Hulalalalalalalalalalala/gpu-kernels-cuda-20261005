# matrixMul - Matrix Multiplication (CUDA Runtime API Version)

## Description

This sample implements matrix multiplication. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.  To illustrate GPU performance for matrix multiply, this sample also shows how to use the CUDA 4.0+ interface for cuBLAS to demonstrate high-performance matrix multiplication.

A is an M x K matrix, B is K x N and C is the row-major M x N single-precision result. M, N and K may be any positive integers — edge tiles are zero-filled so extents that are not multiples of the 32x32 thread-block size (including extents smaller than one block) are computed in full. Products are accumulated in double precision on the GPU so that cancellation-heavy inputs still meet the verification tolerance; every result element is checked against a double-precision CPU reference with an allowed absolute error of `1e-5 + 1e-5*|reference|`, and NaN or infinite values always fail.

```
matrixMul [-device=n] [-wA=W] [-hA=H] [-wB=W] [-hB=H] [-data=constant|mixed]
```

`-data=constant` (the default) fills A with 1.0 and B with 0.01. `-data=mixed` generates deterministic mixed-sign inputs from zero-based indices: `A[r,k] = s[k%4] + ((r+k)%7)*0.125` with `s = [10000, 1, -10000, 0.001]` and `B[k,c] = 1 + ((k+c)%3)*0.0625`, so equal sizes and modes always produce equal inputs. Total device buffer memory is limited to 512 MiB; invalid sizes (zero, negative, non-numeric, overflowing, or `wA != hB`) and unknown data modes are rejected before any allocation.

## Key Concepts

CUDA Runtime API, Linear Algebra

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows, QNX (QNX Safety cross-build with CUDA Safe toolkit)

## Supported CPU Architecture

x86_64, armv7l, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamCreateWithFlags, cudaProfilerStop, cudaMalloc, cudaFree, cudaMallocHost, cudaProfilerStart, cudaEventSynchronize, cudaEventRecord, cudaFreeHost, cudaStreamSynchronize, cudaEventDestroy, cudaEventElapsedTime, cudaMemcpyAsync, cudaEventCreate

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
