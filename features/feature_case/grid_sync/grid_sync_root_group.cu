// ====------ grid_sync_root_group.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cooperative_groups.h>
#include <stdio.h>

__global__ void kernelAdd(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
    g.sync();
    if (idx < N) {
        c[idx] += a[idx];
    }
    g.thread_rank();
    g.block_rank();
    g.num_threads();
    g.num_threads();
    g.size();
}

int main() {
    int N = 1024;
    int gridDim = 256;
    int blockDim = 128;
    int *a, *b, *c;
    int *h_c = (int *)malloc(N * sizeof(int));
    cudaMalloc(&a, N * sizeof(int));
    cudaMalloc(&b, N * sizeof(int));
    cudaMalloc(&c, N * sizeof(int));
        int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }
    cudaMemcpy(a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    void *kernelArgs[] = {
        (void *)&a, (void *)&b, (void *)&c, (void *)&N
    };
    dim3 grid(gridDim, 1, 1);
    dim3 block(blockDim, 1, 1);
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
        cudaLaunchCooperativeKernel((void*)kernelAdd, grid, block, kernelArgs);
    cudaMemcpy(h_c, c, N * sizeof(int), cudaMemcpyDeviceToHost);
    bool result_correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 2 * h_a[i] + h_b[i]) {
            printf("Error: Element c[%d] = %d is incorrect\n", i, h_c[i]);
            result_correct = false;
            break;
        }
        printf("value %d \n", h_c[i]);
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(h_c);
    return 0;
}
