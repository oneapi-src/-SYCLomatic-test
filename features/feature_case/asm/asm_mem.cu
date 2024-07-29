// ====------ asm_mem.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void st(int *a) {
  asm volatile("st.global.s32 [%0], %1;" ::"l"(a), "r"(111));
  asm volatile("st.global.s32 [%0 + 4], %1;" ::"l"(a), "r"(222));
  asm volatile("st.global.s32 [%0 + 8], %1;" ::"l"(a), "r"(333));
  asm volatile("st.global.s32 [%0 + 12], %1;" ::"l"(a), "r"(444));
}

bool test_store() {
  int *d_arr = nullptr;
  cudaMalloc(&d_arr, sizeof(int) * 4);
  st<<<1, 1>>>(d_arr);
  cudaStreamSynchronize(0);
  int h_arr[4], exp[] = {111, 222, 333, 444};
  cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  for (int i = 0; i < 4; ++i)
    if (h_arr[i] != exp[i])
      return false;
  return true;
}

__global__ void ld(int *arr, int *arr2) {
  int a, b, c, d;
  asm volatile("ld.global.s32 %0, [%1];" : "=r"(a) : "l"(arr));
  asm volatile("ld.global.s32 %0, [%1 + 4];" : "=r"(b) : "l"(arr));
  asm volatile("ld.global.s32 %0, [%1 + 8];" : "=r"(c) : "l"(arr));
  asm volatile("ld.global.s32 %0, [%1 + 12];" : "=r"(d) : "l"(arr));
  asm volatile("st.global.s32 [%0], %1;" ::"l"(arr2), "r"(a));
  asm volatile("st.global.s32 [%0 + 4], %1;" ::"l"(arr2), "r"(b));
  asm volatile("st.global.s32 [%0 + 8], %1;" ::"l"(arr2), "r"(c));
  asm volatile("st.global.s32 [%0 + 12], %1;" ::"l"(arr2), "r"(d));
}

bool test_load() {
  int h_arr[4], exp[] = {111, 222, 333, 444};
  int *d_arr = nullptr, *d_arr2 = nullptr;
  cudaMalloc(&d_arr, sizeof(int) * 4);
  cudaMalloc(&d_arr2, sizeof(int) * 4);
  cudaMemcpy(d_arr, exp, sizeof(exp), cudaMemcpyHostToDevice);
  ld<<<1, 1>>>(d_arr, d_arr2);
  cudaStreamSynchronize(0);
  cudaMemcpy(h_arr, d_arr2, sizeof(h_arr), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; ++i)
    if (h_arr[i] != exp[i])
      return false;
  return true;
}

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

int main() {
  TEST(test_store);
  TEST(test_load);

  return 0;
}
