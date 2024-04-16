// ====------ asm_vinst.cu --------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------===//

#include <stdio.h>
#include <stdint.h>

#define CHECK(ID, S, CMP)                                                      \
  {                                                                            \
    S;                                                                         \
    if (!(CMP)) {                                                              \
      return ID;                                                               \
    }                                                                          \
  }

// clang-format off
__device__ int vadd() {
  int i, b = 4, c = 5, d = 6;
  unsigned u;
  CHECK(1, asm("vadd.s32.u32.s32 %0, %1, %2;" : "=r"(i) : "r"(3), "r"(4)), i == 7);
  CHECK(2, asm("vadd.u32.u32.s32 %0, %1, %2;" : "=r"(u) : "r"(b), "r"(c)), u == 9);
  CHECK(3, asm("vadd.s32.u32.s32.sat %0, %1, %2;" : "=r"(i) : "r"(b), "r"(INT_MAX)), i == INT_MAX);
  // TODO: Need to keep the same behavior with asm.
  // CHECK(4, asm("vadd.u32.u32.s32.sat %0, %1, %2;" : "=r"(u) : "r"(UINT_MAX), "r"(INT_MAX)), u == 0x7FFFFFFEull);
  CHECK(5, asm("vadd.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-20), "r"(d)), i == -10);
  CHECK(6, asm("vadd.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(c), "r"(-20)), -20);
  CHECK(7, asm("vadd.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-33), "r"(9)), i == 9);

  return 0;
}

__device__ int vsub() {
  int i, b = 4, c = 5, d = 6;
  unsigned u;
  CHECK(1, asm("vsub.s32.u32.s32 %0, %1, %2;" : "=r"(i) : "r"(3), "r"(4)), i == -1);
  CHECK(2, asm("vsub.u32.u32.s32 %0, %1, %2;" : "=r"(u) : "r"(c), "r"(b)), u == 1);
  CHECK(3, asm("vsub.s32.u32.s32.sat %0, %1, %2;" : "=r"(i) : "r"(10), "r"(INT_MIN)), i == INT_MAX);
  CHECK(4, asm("vsub.u32.u32.s32.sat %0, %1, %2;" : "=r"(u) : "r"(0), "r"(1)), u == 0);
  CHECK(5, asm("vsub.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-20), "r"(d)), i == 30);
  CHECK(6, asm("vsub.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(c), "r"(-20)), -20);
  CHECK(7, asm("vsub.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-33), "r"(9)), i == 37);

  return 0;
}

__device__ int vabsdiff() {
  int i, b = 4, c = 5, d = 6;
  unsigned u;
  CHECK(1, asm("vabsdiff.s32.u32.s32 %0, %1, %2;" : "=r"(i) : "r"(3), "r"(4)), i == 1);
  CHECK(2, asm("vabsdiff.u32.u32.s32 %0, %1, %2;" : "=r"(u) : "r"(c), "r"(b)), u == 1);
  CHECK(3, asm("vabsdiff.s32.u32.s32.sat %0, %1, %2;" : "=r"(i) : "r"(10), "r"(INT_MIN)), i == INT_MAX);
  CHECK(4, asm("vabsdiff.u32.u32.s32.sat %0, %1, %2;" : "=r"(u) : "r"(0), "r"(1)), u == 1);
  CHECK(5, asm("vabsdiff.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-20), "r"(d)), i == 30);
  CHECK(6, asm("vabsdiff.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(c), "r"(-20)), -20);
  CHECK(7, asm("vabsdiff.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-33), "r"(9)), i == 37);

  return 0;
}

__device__ int vmin() {
  int i, b = 4, c = 5, d = 6;
  unsigned u;
  CHECK(1, asm("vmin.s32.u32.s32 %0, %1, %2;" : "=r"(i) : "r"(3), "r"(4)), i == 3);
  CHECK(2, asm("vmin.u32.u32.s32 %0, %1, %2;" : "=r"(u) : "r"(c), "r"(b)), u == 4);
  CHECK(3, asm("vmin.s32.u32.s32.sat %0, %1, %2;" : "=r"(i) : "r"(UINT_MAX), "r"(1)), i == 1);
  CHECK(4, asm("vmin.u32.u32.s32.sat %0, %1, %2;" : "=r"(u) : "r"(10), "r"(-1)), u == 0);
  CHECK(5, asm("vmin.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-20), "r"(d)), i == -14);
  CHECK(6, asm("vmin.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(c), "r"(-20)), -20);
  CHECK(7, asm("vmin.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-33), "r"(9)), i == 9);

  return 0;
}

__device__ int vmax() {
  int i, b = 4, c = 5, d = 6;
  unsigned u;
  CHECK(1, asm("vmax.s32.u32.s32 %0, %1, %2;" : "=r"(i) : "r"(3), "r"(4)), i == 4);
  CHECK(2, asm("vmax.u32.u32.s32 %0, %1, %2;" : "=r"(u) : "r"(c), "r"(b)), u == 5);
  CHECK(3, asm("vmax.s32.u32.s32.sat %0, %1, %2;" : "=r"(i) : "r"(UINT_MAX), "r"(1)), i == INT_MAX);
  CHECK(4, asm("vmax.u32.u32.s32.sat %0, %1, %2;" : "=r"(u) : "r"(UINT_MAX), "r"(1)), u == UINT_MAX);
  CHECK(5, asm("vmax.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-20), "r"(d)), i == 10);
  CHECK(6, asm("vmax.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(c), "r"(-20)), -20);
  CHECK(7, asm("vmax.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(i) : "r"(b), "r"(-33), "r"(9)), i == 9);

  return 0;
}

__device__ int vshl() {
  int i;
  unsigned u;
  CHECK(1, asm("vshl.s32.s32.u32.clamp %0, %1, %2;" : "=r"(i) : "r"(1), "r"(4)), i == 16);
  CHECK(2, asm("vshl.s32.u32.u32.clamp %0, %1, %2;" : "=r"(i) : "r"(1), "r"(33)), i == 0);
  CHECK(3, asm("vshl.u32.s32.u32.clamp %0, %1, %2;" : "=r"(u) : "r"(1), "r"(32)), u == 0);
  CHECK(4, asm("vshl.u32.u32.u32.clamp %0, %1, %2;" : "=r"(u) : "r"(1), "r"(2)), u == 4);
  CHECK(5, asm("vshl.s32.s32.u32.wrap %0, %1, %2;" : "=r"(i) : "r"(1), "r"(4)), i == 16);
  CHECK(6, asm("vshl.s32.u32.u32.wrap %0, %1, %2;" : "=r"(i) : "r"(1), "r"(33)), i == 2);
  CHECK(7, asm("vshl.u32.s32.u32.wrap %0, %1, %2;" : "=r"(u) : "r"(1), "r"(32)), u == 1);
  CHECK(8, asm("vshl.u32.u32.u32.wrap %0, %1, %2;" : "=r"(u) : "r"(1), "r"(2)), u == 4);
  return 0;
}

__device__ int vshr() {
  int i;
  unsigned u;
  CHECK(1, asm("vshr.s32.s32.u32.clamp %0, %1, %2;" : "=r"(i) : "r"(32), "r"(4)), i == 2);
  CHECK(2, asm("vshr.s32.u32.u32.clamp %0, %1, %2;" : "=r"(i) : "r"(1), "r"(33)), i == 0);
  CHECK(3, asm("vshr.u32.s32.u32.clamp %0, %1, %2;" : "=r"(u) : "r"(1), "r"(32)), u == 0);
  CHECK(4, asm("vshr.u32.u32.u32.clamp %0, %1, %2;" : "=r"(u) : "r"(3), "r"(2)), u == 0);
  CHECK(5, asm("vshr.s32.s32.u32.wrap %0, %1, %2;" : "=r"(i) : "r"(32), "r"(4)), i == 2);
  CHECK(6, asm("vshr.s32.u32.u32.wrap %0, %1, %2;" : "=r"(i) : "r"(1), "r"(33)), i == 0);
  CHECK(7, asm("vshr.u32.s32.u32.wrap %0, %1, %2;" : "=r"(u) : "r"(1), "r"(32)), u == 1);
  CHECK(8, asm("vshr.u32.u32.u32.wrap %0, %1, %2;" : "=r"(u) : "r"(32), "r"(2)), u == 8);
  return 0;
}

// clang-format on

__global__ void test(int *ec) {
  {
    int res = vadd();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vsub();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vabsdiff();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vmin();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vmax();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vshl();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
  {
    int res = vshr();
    if (res != 0) {
      *ec = res;
      return;
    }
  }
}

int main() {
  int *ec = nullptr;
  cudaMallocManaged(&ec, sizeof(int));
  *ec = 0;
  test<<<1, 1>>>(ec);
  cudaDeviceSynchronize();
  if (*ec != 0) {
    printf("Test failed %d\n", *ec);
  } else {
    printf("Test pass\n");
  }
  return 0;
}
