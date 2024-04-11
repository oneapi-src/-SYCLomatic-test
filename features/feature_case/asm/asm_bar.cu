// ====------ asm_bar.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>

__device__ void bar(int *arr, int *brr) {
  arr[threadIdx.x] = threadIdx.x + 10;
  if (threadIdx.x % 2 == 0) {
    for (int i = 0; i < 1000; ++i)
      arr[threadIdx.x] += arr[threadIdx.x] - 1 * arr[threadIdx.x] - 3;
    if (arr[threadIdx.x] < 0)
      arr[threadIdx.x] = 0;
  }

  asm volatile ("bar.warp.sync %0;" :: "r"(0b1010101010));
  if (threadIdx.x == 1) {
    for (int i = 0; i < 10; ++i) {
      brr[i] = arr[i];
    }
  }
}

__global__ void kernel(int *arr, int *brr) {
  bar(arr, brr);
}

int main() {
  int *arr, *brr;
  cudaMallocManaged(&arr, sizeof(int) * 10);
  cudaMemset(arr, 0, sizeof(int) * 10);
  cudaMallocManaged(&brr, sizeof(int) * 10);
  cudaMemset(brr, 0, sizeof(int) * 10);

  kernel<<<1, 10>>>(arr, brr);
  cudaDeviceSynchronize();
  cudaFree(arr);
  int res = 0;
  for (int i = 1; i < 10; i+= 2)
    if (brr[i] != i + 10 || brr[i - 1] != 0)
      res = 1;
  cudaFree(brr);
  return res;
}
