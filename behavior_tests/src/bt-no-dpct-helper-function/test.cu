// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

void foo() {
    float *d_A, *d_B, *d_C;
	int vector_size = 100;
    cudaMalloc(&d_A, vector_size*sizeof(float));
    cudaMalloc(&d_B, vector_size*sizeof(float));
    cudaMalloc(&d_C, vector_size*sizeof(float));

    VectorAddKernel<<<1, vector_size>>>(d_A, d_B, d_C);
}
