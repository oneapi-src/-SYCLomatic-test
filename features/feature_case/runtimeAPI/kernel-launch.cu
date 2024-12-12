// ====------ kernel-launch.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main(void)
{
    float *d_A, *d_B, *d_C;
	
    cudaMalloc(&d_A, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_B, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_C, VECTOR_SIZE*sizeof(float));
    
    void **args = (void **)malloc(sizeof(float **) * 3);
    args[0] = &d_A;
    args[1] = &d_B;
    args[2] = &d_C;

    int threadsPerBlock = VECTOR_SIZE;
    int blocksPerGrid = 1;

    cudaLaunchKernel((const void *)VectorAddKernel, blocksPerGrid, threadsPerBlock, args, 0, 0);    

    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, 0);

    cudaError_t ret = cudaEventQuery(e);

    while(ret != cudaSuccess)
      ret = cudaEventQuery(e);

    cudaEventDestroy(e);

    float Result[VECTOR_SIZE] = { };
    cudaMemcpy(Result, d_C, VECTOR_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(args);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%f ", Result[i]);    
    }

    return 0;
}

