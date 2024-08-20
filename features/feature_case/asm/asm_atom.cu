// ====------ asm_atom.cu ---------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>

__global__ void atom(int *a, int *res) {
  asm volatile ("atom.global.s32.add %0, [%1], %2;" : "=r"(res[0]) : "l"(a), "r"(3));
  asm volatile ("atom.global.s32.min %0, [%1], %2;" : "=r"(res[1]) : "l"(a + 1), "r"(1));
  asm volatile ("atom.global.s32.max %0, [%1], %2;" : "=r"(res[2]) : "l"(a + 2), "r"(6));
}

int main() {
  int *d_arr, *d_res;
  cudaMallocManaged(&d_arr, sizeof(int) * 3);
  cudaMallocManaged(&d_res, sizeof(int) * 3);
  d_arr[0] = 1;
  d_arr[1] = 2;
  d_arr[2] = 3;
  atom<<<1, 1>>>(d_arr, d_res);
  cudaDeviceSynchronize();
  bool fail = false;
  if (d_res[0] != 1 || d_res[1] != 2 || d_res[2] != 3)
    fail = true;
  if (d_arr[0] != 1 || d_arr[1] != 2 || d_arr[2] != 3)
    fail = true;
  cudaFree(d_arr);
  return !fail;
}
