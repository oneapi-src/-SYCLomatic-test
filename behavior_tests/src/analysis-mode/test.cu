// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

__global__ void kernel(int *a) {
    *a = clock64();
    __syncthreads();
    int b = *a;
}

void foo() {
    int *a;
    size_t b, c;
    cudaMemGetInfo(&b, &c);
    cudaMalloc(&a, sizeof(int));
    cudaFree(a);
}