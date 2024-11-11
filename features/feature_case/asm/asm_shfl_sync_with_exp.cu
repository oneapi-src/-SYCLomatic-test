// ===------- asm_shfl_sync_with_exp.cu ------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include "cuda_runtime.h"
#include <cub/cub.cuh>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__global__ void shfl_sync_bfly(int *data) {
  int tid = threadIdx.x;
  unsigned mask = 0xaaaaaaaa;
  int val = tid;

  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;"
               : "=r"(data[tid])
               : "r"(val), "r"(4), "r"(31), "r"(mask));
}

__global__ void shfl_sync_down(int *data) {
  int tid = threadIdx.x;
  unsigned mask = 0xaaaaaaaa;
  int val = tid;

  asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
               : "=r"(data[tid])
               : "r"(val), "r"(4), "r"(31), "r"(mask));
}

__global__ void shfl_sync_up(int *data) {
  int tid = threadIdx.x;
  unsigned mask = 0xaaaaaaaa;
  int val = tid;

  asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
               : "=r"(data[tid])
               : "r"(val), "r"(4), "r"(0), "r"(mask));
}

__global__ void shfl_sync_idx(int *data) {
  int tid = threadIdx.x;
  unsigned mask = 0xaaaaaaaa;
  int val = tid;

  asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
               : "=r"(data[tid])
               : "r"(val), "r"(4), "r"(31), "r"(mask));
}


bool shfl_sync_bfly_test() {
  int *data;
  cudaMallocManaged(&data, sizeof(int) * 32);
  shfl_sync_bfly<<<1, 32>>>(data);
  cudaDeviceSynchronize();

  int refer[32] = {4,  5,  6,  7,  0,  1,  2,  3,  12, 13, 14,
                   15, 8,  9,  10, 11, 20, 21, 22, 23, 16, 17,
                   18, 19, 28, 29, 30, 31, 24, 25, 26, 27};

  for (int i = 0; i < 4; ++i)
    if (refer[i] != data[i])
      return false;
  return true;
}

bool shfl_sync_down_test() {
  int *data;
  cudaMallocManaged(&data, sizeof(int) * 32);
  shfl_sync_bfly<<<1, 32>>>(data);
  cudaDeviceSynchronize();

  int refer[32] = {4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 31, 28, 29, 30, 31};

  for (int i = 0; i < 4; ++i)
    if (refer[i] != data[i])
      return false;
  return true;
}



bool shfl_sync_up_test() {
  int *data;
  cudaMallocManaged(&data, sizeof(int) * 32);
  shfl_sync_up<<<1, 32>>>(data);
  cudaDeviceSynchronize();

  int refer[32] = {0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,
                   7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

  for (int i = 0; i < 4; ++i)
    if (refer[i] != data[i])
      return false;
  return true;
}

bool shfl_sync_idx_test() {
  int *data;
  cudaMallocManaged(&data, sizeof(int) * 32);
  shfl_sync_idx<<<1, 32>>>(data);
  cudaDeviceSynchronize();

  int refer[32] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

  for (int i = 0; i < 4; ++i)
    if (refer[i] != data[i])
      return false;
  return true;
}

int main() {
  TEST(shfl_sync_bfly_test);
  TEST(shfl_sync_down_test);
  TEST(shfl_sync_up_test);
  TEST(shfl_sync_idx_test);

  return 0;
}
