// ====------ asm_optimize.cu ------------------------------ *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test1(int *ec) {
  int a = 0, b = 1;
  asm("mov.s32 %0, %1;" : "=r"(a) : "r"(b));
  *ec = a == 1;
}

__global__ void test2(int *ec) {
#define TEST2(X, Y)                                                            \
  asm("{\n\t"                                                                  \
      " .reg .pred p;\n\t"                                                     \
      " setp.eq.s32 p, %1, 34;\n\t"                                            \
      " @p mov.s32 %0, 1;\n\t"                                                 \
      "}"                                                                      \
      : "+r"(Y)                                                                \
      : "r"(X))
  int x = 34;
  int y = 0;
  TEST2(x, y);
  *ec = y == 1;
}

__global__ void test3(int *ec) {
#define TEST3(V)                                                               \
  asm("{\n\t"                                                                  \
      " .reg .u32 p<10>;\n\t"                                                  \
      " mov.u32 p0, 2013;\n\t"                                                 \
      " mov.u32 p1, p0;\n\t"                                                   \
      " mov.u32 p2, p1;\n\t"                                                   \
      " mov.u32 p3, p2;\n\t"                                                   \
      " mov.u32 p4, p3;\n\t"                                                   \
      " mov.u32 p5, p4;\n\t"                                                   \
      " mov.u32 p6, p5;\n\t"                                                   \
      " mov.u32 p7, p6;\n\t"                                                   \
      " mov.u32 p8, p7;\n\t"                                                   \
      " mov.u32 p9, p8;\n\t"                                                   \
      " mov.u32 %0, p9;\n\t"                                                   \
      "}"                                                                      \
      : "=r"(V))
  int x = 34;
  TEST3(x);
  *ec = x == 2013;
}

__global__ void test4(int *ec) {
#define TEST4(X, Y)                                                            \
  {                                                                            \
    X = 35;                                                                    \
    asm("{\n\t"                                                                \
        " .reg .pred p;\n\t"                                                   \
        " setp.eq.s32 p, %1, 35;\n\t"                                          \
        " @p mov.s32 %0, 1;\n\t"                                               \
        "}"                                                                    \
        : "+r"(Y)                                                              \
        : "r"(X));                                                             \
    X = X * X;                                                                 \
  }
  int x = 34;
  int y = 0;
  TEST4(x, y);
  *ec = x == 1225 && y == 1;
}

#define MACRO(ID, CMP, S)                                                      \
  {                                                                            \
    S;                                                                         \
    if (!(CMP)) {                                                              \
      return ID;                                                               \
    }                                                                          \
  }

__device__ int test5_device() {
  int s32 = 0, s32x = 1;
  MACRO(7, s32 == 1,
        asm("add.s32.sat %0, %1, %2;"
            : "=r"(s32)
            : "r"(s32x), "r"(0)));
  return 0;
}

__global__ void test5(int *ec) { *ec = test5_device() == 0; }

#define CHECK(fn)                                                              \
  {                                                                            \
    fn<<<1, 1, 1>>>(ec);                                                       \
    cudaDeviceSynchronize();                                                   \
    if (!*ec) {                                                                \
      printf(#fn " test failed\n");                                            \
      res = false;                                                             \
    }                                                                          \
  }

int main() {
  bool res = true;
  int *ec;
  cudaMallocManaged(&ec, sizeof(int));
  CHECK(test1);
  CHECK(test2);
  CHECK(test3);
  CHECK(test4);
  CHECK(test5);
  cudaFree(ec);

  return !res;
}
