// ====------ asm.cu -------------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------===//
//
// This file defines the test cases to check inline device asm parser.
//
// (1) Floating point/integer constant.
// (2) Binary/Unary/Conditional operators and paren expressions.
// (3) Compound statements.
// (4) Conditional instructions.
// (5) Instructions(mov, setp, and lop3).
// (6) Builtin registers.
//
// Usually, we check the result of inline asm statement to ensure that the
// migrated programe has the same behavior with the inline asmstatement.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <map>

#define EPS (1e-6)

template <typename F, typename I> union U {
  F f;
  I i;
};

__global__ void floating_point(int *ec) {
  float f32;
  double f64;
  asm("mov.f32 %0, 0F3f800000;" : "=f"(f32));
  if (f32 != U<float, uint32_t>{.i = 0x3f800000U}.f) {
    *ec = 1;
    return;
  }

  asm("mov.f32 %0, 0f3f800000;" : "=f"(f32));
  if (f32 != U<float, uint32_t>{.i = 0x3f800000U}.f) {
    *ec = 2;
    return;
  }

  asm("mov.f64 %0, 0D40091EB851EB851F;" : "=d"(f64));
  if (f64 != U<double, uint64_t>{.i = 0x40091EB851EB851FULL}.f) {
    *ec = 3;
    return;
  }

  asm("mov.f64 %0, 0d40091EB851EB851F;" : "=d"(f64));
  if (f64 != U<double, uint64_t>{.i = 0x40091EB851EB851FULL}.f) {
    *ec = 4;
    return;
  }

  asm("mov.f64 %0, 0.1;" : "=d"(f64));
  if (std::abs(f64 - 0.1) > EPS) {
    *ec = 5;
    return;
  }

  asm("mov.f64 %0, 1e10;" : "=d"(f64));
  if (std::abs(f64 - 1e10) > EPS) {
    *ec = 6;
    return;
  }
  *ec = 0;
}

__global__ void integer_literal(int *ec) {
  uint32_t u32;

  asm("mov.u32 %0, 123;" : "=r"(u32));
  if (u32 != 123) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 0123;" : "=r"(u32));
  if (u32 != 0123) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 0xFF;" : "=r"(u32));
  if (u32 != 0xFF) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 0b101;" : "=r"(u32));
  if (u32 != 0b101) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 123U;" : "=r"(u32));
  if (u32 != 123U) {
    *ec = 1;
    return;
  }

  *ec = 0;
}

__global__ void expression(int *ec) {
  uint32_t u32;

  asm("mov.u32 %0, 123 + 123;" : "=r"(u32));
  if (u32 != 123 + 123) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 123 + 123 * 7;" : "=r"(u32));
  if (u32 != 123 + 123 * 7) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 3 & 1;" : "=r"(u32));
  if (u32 != (3 & 1)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 3 == 1;" : "=r"(u32));
  if (u32 != (3 == 1)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, ~7;" : "=r"(u32));
  if (u32 != (~7)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, !1;" : "=r"(u32));
  if (u32 != (!1)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, -4;" : "=r"(u32));
  if (u32 != uint32_t(-4)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 8 ^ 8;" : "=r"(u32));
  if (u32 != (8 ^ 8)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 1 ? 3 : 2;" : "=r"(u32));
  if (u32 != (1 ? 3 : 2)) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, (3 + 7) * 10;" : "=r"(u32));
  if (u32 != (3 + 7) * 10) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, (3 + 7) * 10 / 5;" : "=r"(u32));
  if (u32 != (3 + 7) * 10 / 5) {
    *ec = 1;
    return;
  }

  asm("mov.u32 %0, 99 %% 10;" : "=r"(u32));
  if (u32 != 99 % 10) {
    *ec = 1;
    return;
  }

  *ec = 0;
}

__global__ void declaration(int *ec) {
#define COND(X, Y)                                                             \
  asm("{\n\t"                                                                  \
      " .reg .pred p;\n\t"                                                     \
      " setp.eq.s32 p, %1, 34;\n\t"                                            \
      " @p mov.s32 %0, 1;\n\t"                                                 \
      "}"                                                                      \
      : "+r"(Y)                                                                \
      : "r"(X))

#define ARR(V)                                                                 \
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
      " mov.u32 %0, p8;\n\t"                                                   \
      "}"                                                                      \
      : "=r"(V))

  // This variable was used to test compound statement in inline asm,
  // if the '{...}' was not correctly emitted, then, it will cause
  // variable redefinition.
  [[maybe_unused]] int p;
  int x = 34;
  int y = 0;
  COND(x, y);
  if (y != 1) {
    *ec = 1;
    return;
  }

  x = 33;
  y = 0;
  COND(x, y);
  if (y == 1) {
    *ec = 2;
    return;
  }

  x = 0;
  ARR(x);
  if (x != 2013) {
    *ec = 3;
    return;
  }

  *ec = 0;
}

__global__ void builtin(int *ec, int *laneids, int *warpszs, int *warpids) {
  unsigned laneid, warp_size, warpid;
  unsigned tid =
      ((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) +
      (threadIdx.x + (threadIdx.y * blockDim.x));
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  asm volatile("mov.u32 %0, WARP_SZ;" : "=r"(warp_size));
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  unsigned laneid2 = (threadIdx.x & (warpSize - 1));
  if (laneid != laneid2) {
    *ec = 1;
    return;
  }
  laneids[tid] = laneid;
  warpszs[tid] = warp_size;
  warpids[tid] = warpid;
  if (tid == 0) {
    *ec = 0;
  }
  // printf("laneid=%u\n", laneid);
}

__global__ void setp(int *ec) {
  int32_t i32;
  uint32_t u32;
  float f32;

  i32 = 34;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.eq.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 != 1) {
    *ec = 1;
    return;
  }

  i32 = 34;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ne.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 == 1) {
    *ec = 2;
    return;
  }

  i32 = 31;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.lt.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 != 1) {
    *ec = 3;
    return;
  }

  i32 = 34;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.le.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 != 1) {
    *ec = 4;
    return;
  }

  i32 = 35;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.gt.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 != 1) {
    *ec = 5;
    return;
  }

  i32 = 34;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ge.s32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(i32));

  if (i32 != 1) {
    *ec = 6;
    return;
  }

  u32 = 34;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ls.u32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(u32));

  if (u32 != 1) {
    *ec = 7;
    return;
  }

  u32 = 33;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.lo.u32 p, %0, 34;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "+r"(u32));

  if (u32 != 1) {
    *ec = 8;
    return;
  }

  u32 = 36;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.hs.u32 p, %0, 34;\n\t"
      " @p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(u32));

  if (u32 != 1) {
    *ec = 9;
    return;
  }

  u32 = 35;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.hi.u32 p, %0, 34;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "+r"(u32));

  if (u32 != 1) {
    *ec = 10;
    return;
  }

  // eq
  f32 = 0.1;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.eq.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 11;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.eq.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 12;
    printf("return ???\n");
    return;
  }

  // ne
  f32 = 0.2;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ne.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 13;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ne.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 14;
    return;
  }

  // lt
  f32 = 0.2;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.lt.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 15;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.lt.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 16;
    return;
  }

  // le
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.le.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 17;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.le.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 18;
    return;
  }

  // gt
  f32 = 0.6;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.gt.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 19;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.gt.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 20;
    return;
  }

  // ge
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ge.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 21;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ge.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 22;
    return;
  }

  // equ
  f32 = 0.1;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.equ.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 23;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.equ.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 24;
    return;
  }

  // neu
  u32 = 0;
  f32 = 0.2;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.neu.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 25;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.neu.f32 p, %1, 0.1;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 26;
    return;
  }

  // ltu
  f32 = 0.2;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ltu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 27;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.ltu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 28;
    return;
  }

  // leu
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.leu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 29;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.leu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 30;
    return;
  }

  // gtu
  f32 = 0.6;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.gtu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 31;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.gtu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 32;
    return;
  }

  // geu
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.geu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 33;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.geu.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 34;
    return;
  }

  // num
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.num.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 35;
    return;
  }

  f32 = 1.0;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.num.f32 p, %1, 0f7FC00000;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 36;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.num.f32 p, %1, 0f7FC00000;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 37;
    return;
  }

  // nan
  f32 = 0.5;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.nan.f32 p, %1, 0.5;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 == 1) {
    *ec = 38;
    return;
  }

  f32 = 1.0;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.nan.f32 p, %1, 0f7FC00000;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 39;
    return;
  }

  f32 = NAN;
  u32 = 0;
  asm("{\n\t"
      " .reg .pred p;\n\t"
      " setp.nan.f32 p, %1, 0f7FC00000;\n\t"
      " @p mov.u32 %0, 1;\n\t"
      "}"
      : "=r"(u32)
      : "f"(f32));

  if (u32 != 1) {
    *ec = 40;
    return;
  }
}

// clang-format off
__device__ void reference_of_lop3(uint32_t &R, uint32_t A, uint32_t B, uint32_t C, uint32_t D) {
  switch (D) {
  case 0: R = 0; break;
  case 1: R = (~A & ~B & ~C); break;
  case 2: R = (~A & ~B & C); break;
  case 3: R = (~A & ~B & ~C) | (~A & ~B & C); break;
  case 4: R = (~A & B & ~C); break;
  case 5: R = (~A & ~B & ~C) | (~A & B & ~C); break;
  case 6: R = (~A & ~B & C) | (~A & B & ~C); break;
  case 7: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C); break;
  case 8: R = (~A & B & C); break;
  case 9: R = (~A & ~B & ~C) | (~A & B & C); break;
  case 10: R = (~A & ~B & C) | (~A & B & C); break;
  case 11: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C); break;
  case 12: R = (~A & B & ~C) | (~A & B & C); break;
  case 13: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C); break;
  case 14: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C); break;
  case 15: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C); break;
  case 16: R = (A & ~B & ~C); break;
  case 17: R = (~A & ~B & ~C) | (A & ~B & ~C); break;
  case 18: R = (~A & ~B & C) | (A & ~B & ~C); break;
  case 19: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C); break;
  case 20: R = (~A & B & ~C) | (A & ~B & ~C); break;
  case 21: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 22: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 23: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 24: R = (~A & B & C) | (A & ~B & ~C); break;
  case 25: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 26: R = (A & B | C) ^ A; break;
  case 27: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C); break;
  case 28: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 29: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 30: R = A ^ (B | C); break;
  case 31: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 32: R = (A & ~B & C); break;
  case 33: R = (~A & ~B & ~C) | (A & ~B & C); break;
  case 34: R = (~A & ~B & C) | (A & ~B & C); break;
  case 35: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C); break;
  case 36: R = (~A & B & ~C) | (A & ~B & C); break;
  case 37: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C); break;
  case 38: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C); break;
  case 39: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C); break;
  case 40: R = (~A & B & C) | (A & ~B & C); break;
  case 41: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 42: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C); break;
  case 43: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C); break;
  case 44: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 45: R = ~A ^ (~B & C); break;
  case 46: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 47: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 48: R = (A & ~B & ~C) | (A & ~B & C); break;
  case 49: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 50: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 51: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 52: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 53: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 54: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 55: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 56: R = (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 57: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 58: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 59: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 60: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 61: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 62: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 63: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 64: R = A & B & ~C; break;
  case 65: R = (~A & ~B & ~C) | (A & B & ~C); break;
  case 66: R = (~A & ~B & C) | (A & B & ~C); break;
  case 67: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & ~C); break;
  case 68: R = (~A & B & ~C) | (A & B & ~C); break;
  case 69: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & ~C); break;
  case 70: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C); break;
  case 71: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C); break;
  case 72: R = (~A & B & C) | (A & B & ~C); break;
  case 73: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 74: R = (~A & ~B & C) | (~A & B & C) | (A & B & ~C); break;
  case 75: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & ~C); break;
  case 76: R = (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 77: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 78: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 79: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 80: R = (A & ~B & ~C) | (A & B & ~C); break;
  case 81: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 82: R = (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 83: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 84: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 85: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 86: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 87: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 88: R = (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 89: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 90: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 91: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 92: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 93: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 94: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 95: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 96: R = (A & ~B & C) | (A & B & ~C); break;
  case 97: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 98: R = (~A & ~B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 99: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 100: R = (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 101: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 102: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 103: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 104: R = (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 105: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 106: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 107: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 108: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 109: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 110: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 111: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 112: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 113: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 114: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 115: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 116: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 117: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 118: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 119: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 120: R = A ^ (B & C); break;
  case 121: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 122: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 123: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 124: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 125: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 126: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 127: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 128: R = A & B & C; break;
  case 129: R = (~A & ~B & ~C) | (A & B & C); break;
  case 130: R = (~A & ~B & C) | (A & B & C); break;
  case 131: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & C); break;
  case 132: R = (~A & B & ~C) | (A & B & C); break;
  case 133: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & C); break;
  case 134: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & C); break;
  case 135: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & C); break;
  case 136: R = (~A & B & C) | (A & B & C); break;
  case 137: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & C); break;
  case 138: R = (~A & ~B & C) | (~A & B & C) | (A & B & C); break;
  case 139: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & C); break;
  case 140: R = (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 141: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 142: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 143: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 144: R = (A & ~B & ~C) | (A & B & C); break;
  case 145: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 146: R = (~A & ~B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 147: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 148: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 149: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 150: R = A ^ B ^ C; break;
  case 151: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 152: R = (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 153: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 154: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 155: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 156: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 157: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 158: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 159: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 160: R = (A & ~B & C) | (A & B & C); break;
  case 161: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 162: R = (~A & ~B & C) | (A & ~B & C) | (A & B & C); break;
  case 163: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & C); break;
  case 164: R = (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 165: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 166: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 167: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 168: R = (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 169: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 170: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 171: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 172: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 173: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 174: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 175: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 176: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 177: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 178: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 179: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 180: R = A ^ (B & ~C); break;
  case 181: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 182: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 183: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 184: R = (A ^ (B & (C ^ A))); break;
  case 185: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 186: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 187: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 188: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 189: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 190: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 191: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 192: R = (A & B & ~C) | (A & B & C); break;
  case 193: R = (~A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 194: R = (~A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 195: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 196: R = (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 197: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 198: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 199: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 200: R = (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 201: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 202: R = (~A & ~B & C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 203: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 204: R = (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 205: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 206: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 207: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 208: R = (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 209: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 210: R = A ^ (~B & C); break;
  case 211: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 212: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 213: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 214: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 215: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 216: R = (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 217: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 218: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 219: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 220: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 221: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 222: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 223: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 224: R = (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 225: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 226: R = (~A & ~B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 227: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 228: R = (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 229: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 230: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 231: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 232: R = ((A & (B | C)) | (B & C)); break;
  case 233: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 234: R = (A & B) | C; break;
  case 235: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 236: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 237: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 238: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 239: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 240: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 241: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 242: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 243: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 244: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 245: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 246: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 247: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 248: R = (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 249: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 250: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 251: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 252: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 253: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 254: R = A | B | C; break;
  case 255: R = uint32_t(-1); break;
  default: break;
  }
}

__device__ void asm_lop3(uint32_t &R, uint32_t A, uint32_t B, uint32_t C, uint32_t D) {
  switch (D) {
  case 0: asm("lop3.b32 %0, %1, %2, %3, 0x0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 1: asm("lop3.b32 %0, %1, %2, %3, 0x1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 2: asm("lop3.b32 %0, %1, %2, %3, 0x2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 3: asm("lop3.b32 %0, %1, %2, %3, 0x3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 4: asm("lop3.b32 %0, %1, %2, %3, 0x4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 5: asm("lop3.b32 %0, %1, %2, %3, 0x5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 6: asm("lop3.b32 %0, %1, %2, %3, 0x6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 7: asm("lop3.b32 %0, %1, %2, %3, 0x7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 8: asm("lop3.b32 %0, %1, %2, %3, 0x8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 9: asm("lop3.b32 %0, %1, %2, %3, 0x9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 10: asm("lop3.b32 %0, %1, %2, %3, 0xA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 11: asm("lop3.b32 %0, %1, %2, %3, 0xB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 12: asm("lop3.b32 %0, %1, %2, %3, 0xC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 13: asm("lop3.b32 %0, %1, %2, %3, 0xD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 14: asm("lop3.b32 %0, %1, %2, %3, 0xE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 15: asm("lop3.b32 %0, %1, %2, %3, 0xF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 16: asm("lop3.b32 %0, %1, %2, %3, 0x10;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 17: asm("lop3.b32 %0, %1, %2, %3, 0x11;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 18: asm("lop3.b32 %0, %1, %2, %3, 0x12;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 19: asm("lop3.b32 %0, %1, %2, %3, 0x13;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 20: asm("lop3.b32 %0, %1, %2, %3, 0x14;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 21: asm("lop3.b32 %0, %1, %2, %3, 0x15;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 22: asm("lop3.b32 %0, %1, %2, %3, 0x16;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 23: asm("lop3.b32 %0, %1, %2, %3, 0x17;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 24: asm("lop3.b32 %0, %1, %2, %3, 0x18;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 25: asm("lop3.b32 %0, %1, %2, %3, 0x19;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 26: asm("lop3.b32 %0, %1, %2, %3, 0x1A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 27: asm("lop3.b32 %0, %1, %2, %3, 0x1B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 28: asm("lop3.b32 %0, %1, %2, %3, 0x1C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 29: asm("lop3.b32 %0, %1, %2, %3, 0x1D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 30: asm("lop3.b32 %0, %1, %2, %3, 0x1E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 31: asm("lop3.b32 %0, %1, %2, %3, 0x1F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 32: asm("lop3.b32 %0, %1, %2, %3, 0x20;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 33: asm("lop3.b32 %0, %1, %2, %3, 0x21;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 34: asm("lop3.b32 %0, %1, %2, %3, 0x22;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 35: asm("lop3.b32 %0, %1, %2, %3, 0x23;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 36: asm("lop3.b32 %0, %1, %2, %3, 0x24;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 37: asm("lop3.b32 %0, %1, %2, %3, 0x25;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 38: asm("lop3.b32 %0, %1, %2, %3, 0x26;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 39: asm("lop3.b32 %0, %1, %2, %3, 0x27;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 40: asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 41: asm("lop3.b32 %0, %1, %2, %3, 0x29;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 42: asm("lop3.b32 %0, %1, %2, %3, 0x2A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 43: asm("lop3.b32 %0, %1, %2, %3, 0x2B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 44: asm("lop3.b32 %0, %1, %2, %3, 0x2C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 45: asm("lop3.b32 %0, %1, %2, %3, 0x2D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 46: asm("lop3.b32 %0, %1, %2, %3, 0x2E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 47: asm("lop3.b32 %0, %1, %2, %3, 0x2F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 48: asm("lop3.b32 %0, %1, %2, %3, 0x30;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 49: asm("lop3.b32 %0, %1, %2, %3, 0x31;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 50: asm("lop3.b32 %0, %1, %2, %3, 0x32;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 51: asm("lop3.b32 %0, %1, %2, %3, 0x33;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 52: asm("lop3.b32 %0, %1, %2, %3, 0x34;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 53: asm("lop3.b32 %0, %1, %2, %3, 0x35;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 54: asm("lop3.b32 %0, %1, %2, %3, 0x36;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 55: asm("lop3.b32 %0, %1, %2, %3, 0x37;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 56: asm("lop3.b32 %0, %1, %2, %3, 0x38;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 57: asm("lop3.b32 %0, %1, %2, %3, 0x39;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 58: asm("lop3.b32 %0, %1, %2, %3, 0x3A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 59: asm("lop3.b32 %0, %1, %2, %3, 0x3B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 60: asm("lop3.b32 %0, %1, %2, %3, 0x3C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 61: asm("lop3.b32 %0, %1, %2, %3, 0x3D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 62: asm("lop3.b32 %0, %1, %2, %3, 0x3E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 63: asm("lop3.b32 %0, %1, %2, %3, 0x3F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 64: asm("lop3.b32 %0, %1, %2, %3, 0x40;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 65: asm("lop3.b32 %0, %1, %2, %3, 0x41;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 66: asm("lop3.b32 %0, %1, %2, %3, 0x42;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 67: asm("lop3.b32 %0, %1, %2, %3, 0x43;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 68: asm("lop3.b32 %0, %1, %2, %3, 0x44;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 69: asm("lop3.b32 %0, %1, %2, %3, 0x45;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 70: asm("lop3.b32 %0, %1, %2, %3, 0x46;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 71: asm("lop3.b32 %0, %1, %2, %3, 0x47;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 72: asm("lop3.b32 %0, %1, %2, %3, 0x48;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 73: asm("lop3.b32 %0, %1, %2, %3, 0x49;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 74: asm("lop3.b32 %0, %1, %2, %3, 0x4A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 75: asm("lop3.b32 %0, %1, %2, %3, 0x4B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 76: asm("lop3.b32 %0, %1, %2, %3, 0x4C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 77: asm("lop3.b32 %0, %1, %2, %3, 0x4D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 78: asm("lop3.b32 %0, %1, %2, %3, 0x4E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 79: asm("lop3.b32 %0, %1, %2, %3, 0x4F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 80: asm("lop3.b32 %0, %1, %2, %3, 0x50;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 81: asm("lop3.b32 %0, %1, %2, %3, 0x51;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 82: asm("lop3.b32 %0, %1, %2, %3, 0x52;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 83: asm("lop3.b32 %0, %1, %2, %3, 0x53;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 84: asm("lop3.b32 %0, %1, %2, %3, 0x54;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 85: asm("lop3.b32 %0, %1, %2, %3, 0x55;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 86: asm("lop3.b32 %0, %1, %2, %3, 0x56;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 87: asm("lop3.b32 %0, %1, %2, %3, 0x57;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 88: asm("lop3.b32 %0, %1, %2, %3, 0x58;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 89: asm("lop3.b32 %0, %1, %2, %3, 0x59;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 90: asm("lop3.b32 %0, %1, %2, %3, 0x5A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 91: asm("lop3.b32 %0, %1, %2, %3, 0x5B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 92: asm("lop3.b32 %0, %1, %2, %3, 0x5C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 93: asm("lop3.b32 %0, %1, %2, %3, 0x5D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 94: asm("lop3.b32 %0, %1, %2, %3, 0x5E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 95: asm("lop3.b32 %0, %1, %2, %3, 0x5F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 96: asm("lop3.b32 %0, %1, %2, %3, 0x60;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 97: asm("lop3.b32 %0, %1, %2, %3, 0x61;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 98: asm("lop3.b32 %0, %1, %2, %3, 0x62;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 99: asm("lop3.b32 %0, %1, %2, %3, 0x63;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 100: asm("lop3.b32 %0, %1, %2, %3, 0x64;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 101: asm("lop3.b32 %0, %1, %2, %3, 0x65;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 102: asm("lop3.b32 %0, %1, %2, %3, 0x66;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 103: asm("lop3.b32 %0, %1, %2, %3, 0x67;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 104: asm("lop3.b32 %0, %1, %2, %3, 0x68;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 105: asm("lop3.b32 %0, %1, %2, %3, 0x69;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 106: asm("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 107: asm("lop3.b32 %0, %1, %2, %3, 0x6B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 108: asm("lop3.b32 %0, %1, %2, %3, 0x6C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 109: asm("lop3.b32 %0, %1, %2, %3, 0x6D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 110: asm("lop3.b32 %0, %1, %2, %3, 0x6E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 111: asm("lop3.b32 %0, %1, %2, %3, 0x6F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 112: asm("lop3.b32 %0, %1, %2, %3, 0x70;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 113: asm("lop3.b32 %0, %1, %2, %3, 0x71;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 114: asm("lop3.b32 %0, %1, %2, %3, 0x72;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 115: asm("lop3.b32 %0, %1, %2, %3, 0x73;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 116: asm("lop3.b32 %0, %1, %2, %3, 0x74;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 117: asm("lop3.b32 %0, %1, %2, %3, 0x75;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 118: asm("lop3.b32 %0, %1, %2, %3, 0x76;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 119: asm("lop3.b32 %0, %1, %2, %3, 0x77;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 120: asm("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 121: asm("lop3.b32 %0, %1, %2, %3, 0x79;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 122: asm("lop3.b32 %0, %1, %2, %3, 0x7A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 123: asm("lop3.b32 %0, %1, %2, %3, 0x7B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 124: asm("lop3.b32 %0, %1, %2, %3, 0x7C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 125: asm("lop3.b32 %0, %1, %2, %3, 0x7D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 126: asm("lop3.b32 %0, %1, %2, %3, 0x7E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 127: asm("lop3.b32 %0, %1, %2, %3, 0x7F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 128: asm("lop3.b32 %0, %1, %2, %3, 0x80;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 129: asm("lop3.b32 %0, %1, %2, %3, 0x81;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 130: asm("lop3.b32 %0, %1, %2, %3, 0x82;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 131: asm("lop3.b32 %0, %1, %2, %3, 0x83;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 132: asm("lop3.b32 %0, %1, %2, %3, 0x84;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 133: asm("lop3.b32 %0, %1, %2, %3, 0x85;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 134: asm("lop3.b32 %0, %1, %2, %3, 0x86;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 135: asm("lop3.b32 %0, %1, %2, %3, 0x87;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 136: asm("lop3.b32 %0, %1, %2, %3, 0x88;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 137: asm("lop3.b32 %0, %1, %2, %3, 0x89;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 138: asm("lop3.b32 %0, %1, %2, %3, 0x8A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 139: asm("lop3.b32 %0, %1, %2, %3, 0x8B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 140: asm("lop3.b32 %0, %1, %2, %3, 0x8C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 141: asm("lop3.b32 %0, %1, %2, %3, 0x8D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 142: asm("lop3.b32 %0, %1, %2, %3, 0x8E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 143: asm("lop3.b32 %0, %1, %2, %3, 0x8F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 144: asm("lop3.b32 %0, %1, %2, %3, 0x90;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 145: asm("lop3.b32 %0, %1, %2, %3, 0x91;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 146: asm("lop3.b32 %0, %1, %2, %3, 0x92;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 147: asm("lop3.b32 %0, %1, %2, %3, 0x93;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 148: asm("lop3.b32 %0, %1, %2, %3, 0x94;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 149: asm("lop3.b32 %0, %1, %2, %3, 0x95;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 150: asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 151: asm("lop3.b32 %0, %1, %2, %3, 0x97;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 152: asm("lop3.b32 %0, %1, %2, %3, 0x98;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 153: asm("lop3.b32 %0, %1, %2, %3, 0x99;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 154: asm("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 155: asm("lop3.b32 %0, %1, %2, %3, 0x9B;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 156: asm("lop3.b32 %0, %1, %2, %3, 0x9C;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 157: asm("lop3.b32 %0, %1, %2, %3, 0x9D;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 158: asm("lop3.b32 %0, %1, %2, %3, 0x9E;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 159: asm("lop3.b32 %0, %1, %2, %3, 0x9F;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 160: asm("lop3.b32 %0, %1, %2, %3, 0xA0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 161: asm("lop3.b32 %0, %1, %2, %3, 0xA1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 162: asm("lop3.b32 %0, %1, %2, %3, 0xA2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 163: asm("lop3.b32 %0, %1, %2, %3, 0xA3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 164: asm("lop3.b32 %0, %1, %2, %3, 0xA4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 165: asm("lop3.b32 %0, %1, %2, %3, 0xA5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 166: asm("lop3.b32 %0, %1, %2, %3, 0xA6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 167: asm("lop3.b32 %0, %1, %2, %3, 0xA7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 168: asm("lop3.b32 %0, %1, %2, %3, 0xA8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 169: asm("lop3.b32 %0, %1, %2, %3, 0xA9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 170: asm("lop3.b32 %0, %1, %2, %3, 0xAA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 171: asm("lop3.b32 %0, %1, %2, %3, 0xAB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 172: asm("lop3.b32 %0, %1, %2, %3, 0xAC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 173: asm("lop3.b32 %0, %1, %2, %3, 0xAD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 174: asm("lop3.b32 %0, %1, %2, %3, 0xAE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 175: asm("lop3.b32 %0, %1, %2, %3, 0xAF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 176: asm("lop3.b32 %0, %1, %2, %3, 0xB0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 177: asm("lop3.b32 %0, %1, %2, %3, 0xB1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 178: asm("lop3.b32 %0, %1, %2, %3, 0xB2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 179: asm("lop3.b32 %0, %1, %2, %3, 0xB3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 180: asm("lop3.b32 %0, %1, %2, %3, 0xB4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 181: asm("lop3.b32 %0, %1, %2, %3, 0xB5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 182: asm("lop3.b32 %0, %1, %2, %3, 0xB6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 183: asm("lop3.b32 %0, %1, %2, %3, 0xB7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 184: asm("lop3.b32 %0, %1, %2, %3, 0xB8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 185: asm("lop3.b32 %0, %1, %2, %3, 0xB9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 186: asm("lop3.b32 %0, %1, %2, %3, 0xBA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 187: asm("lop3.b32 %0, %1, %2, %3, 0xBB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 188: asm("lop3.b32 %0, %1, %2, %3, 0xBC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 189: asm("lop3.b32 %0, %1, %2, %3, 0xBD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 190: asm("lop3.b32 %0, %1, %2, %3, 0xBE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 191: asm("lop3.b32 %0, %1, %2, %3, 0xBF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 192: asm("lop3.b32 %0, %1, %2, %3, 0xC0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 193: asm("lop3.b32 %0, %1, %2, %3, 0xC1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 194: asm("lop3.b32 %0, %1, %2, %3, 0xC2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 195: asm("lop3.b32 %0, %1, %2, %3, 0xC3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 196: asm("lop3.b32 %0, %1, %2, %3, 0xC4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 197: asm("lop3.b32 %0, %1, %2, %3, 0xC5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 198: asm("lop3.b32 %0, %1, %2, %3, 0xC6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 199: asm("lop3.b32 %0, %1, %2, %3, 0xC7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 200: asm("lop3.b32 %0, %1, %2, %3, 0xC8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 201: asm("lop3.b32 %0, %1, %2, %3, 0xC9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 202: asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 203: asm("lop3.b32 %0, %1, %2, %3, 0xCB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 204: asm("lop3.b32 %0, %1, %2, %3, 0xCC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 205: asm("lop3.b32 %0, %1, %2, %3, 0xCD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 206: asm("lop3.b32 %0, %1, %2, %3, 0xCE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 207: asm("lop3.b32 %0, %1, %2, %3, 0xCF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 208: asm("lop3.b32 %0, %1, %2, %3, 0xD0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 209: asm("lop3.b32 %0, %1, %2, %3, 0xD1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 210: asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 211: asm("lop3.b32 %0, %1, %2, %3, 0xD3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 212: asm("lop3.b32 %0, %1, %2, %3, 0xD4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 213: asm("lop3.b32 %0, %1, %2, %3, 0xD5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 214: asm("lop3.b32 %0, %1, %2, %3, 0xD6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 215: asm("lop3.b32 %0, %1, %2, %3, 0xD7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 216: asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 217: asm("lop3.b32 %0, %1, %2, %3, 0xD9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 218: asm("lop3.b32 %0, %1, %2, %3, 0xDA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 219: asm("lop3.b32 %0, %1, %2, %3, 0xDB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 220: asm("lop3.b32 %0, %1, %2, %3, 0xDC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 221: asm("lop3.b32 %0, %1, %2, %3, 0xDD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 222: asm("lop3.b32 %0, %1, %2, %3, 0xDE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 223: asm("lop3.b32 %0, %1, %2, %3, 0xDF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 224: asm("lop3.b32 %0, %1, %2, %3, 0xE0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 225: asm("lop3.b32 %0, %1, %2, %3, 0xE1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 226: asm("lop3.b32 %0, %1, %2, %3, 0xE2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 227: asm("lop3.b32 %0, %1, %2, %3, 0xE3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 228: asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 229: asm("lop3.b32 %0, %1, %2, %3, 0xE5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 230: asm("lop3.b32 %0, %1, %2, %3, 0xE6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 231: asm("lop3.b32 %0, %1, %2, %3, 0xE7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 232: asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 233: asm("lop3.b32 %0, %1, %2, %3, 0xE9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 234: asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 235: asm("lop3.b32 %0, %1, %2, %3, 0xEB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 236: asm("lop3.b32 %0, %1, %2, %3, 0xEC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 237: asm("lop3.b32 %0, %1, %2, %3, 0xED;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 238: asm("lop3.b32 %0, %1, %2, %3, 0xEE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 239: asm("lop3.b32 %0, %1, %2, %3, 0xEF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 240: asm("lop3.b32 %0, %1, %2, %3, 0xF0;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 241: asm("lop3.b32 %0, %1, %2, %3, 0xF1;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 242: asm("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 243: asm("lop3.b32 %0, %1, %2, %3, 0xF3;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 244: asm("lop3.b32 %0, %1, %2, %3, 0xF4;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 245: asm("lop3.b32 %0, %1, %2, %3, 0xF5;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 246: asm("lop3.b32 %0, %1, %2, %3, 0xF6;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 247: asm("lop3.b32 %0, %1, %2, %3, 0xF7;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 248: asm("lop3.b32 %0, %1, %2, %3, 0xF8;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 249: asm("lop3.b32 %0, %1, %2, %3, 0xF9;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 250: asm("lop3.b32 %0, %1, %2, %3, 0xFA;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 251: asm("lop3.b32 %0, %1, %2, %3, 0xFB;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 252: asm("lop3.b32 %0, %1, %2, %3, 0xFC;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 253: asm("lop3.b32 %0, %1, %2, %3, 0xFD;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 254: asm("lop3.b32 %0, %1, %2, %3, 0xFE;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  case 255: asm("lop3.b32 %0, %1, %2, %3, 0xFF;" : "=r"(R) : "r"(A), "r"(B), "r"(C)); break;
  }
}

// clang-format on

__global__ void lop3(int *ec) {
  uint32_t X, Y, A = 1, B = 2, C = 3, D;
  for (D = 0; D < 256; ++D) {
    reference_of_lop3(X, A, B, C, D);
    asm_lop3(Y, A, B, C, D);
    if (X != Y) {
      *ec = D;
      return;
    }
  }
  *ec = 0;
}

template <int lut, typename T> __device__ inline T lop3(T a, T b, T c) {
  T res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

__global__ void lop3_template(int *ec) {
  uint32_t X, Y, A = 1, B = 2, C = 3;
  reference_of_lop3(X, A, B, C, 0x8);
  Y = lop3<0x8>(A, B, C);
  if (X != Y) {
    *ec = Y;
    return;
  }

  *ec = 0;
}

int main() {
  int ret = 0;
  int *d_ec = nullptr;
  cudaMalloc(&d_ec, sizeof(int));

  auto wait_and_check = [&](const char *case_name) {
    cudaDeviceSynchronize();
    int ec = 0;
    cudaMemcpy(&ec, d_ec, sizeof(int), cudaMemcpyDeviceToHost);
    if (ec != 0)
      printf("Test %s failed: return code = %d\n", case_name, ec);
    ret = ret || ec;
  };

  floating_point<<<1, 1>>>(d_ec);
  wait_and_check("floating point");

  integer_literal<<<1, 1>>>(d_ec);
  wait_and_check("integer literal");

  expression<<<1, 1>>>(d_ec);
  wait_and_check("expression");

  declaration<<<1, 1>>>(d_ec);
  wait_and_check("declaration");

  int *d_warpids, *d_warpszs, *d_laneids;
  cudaMalloc(&d_laneids, sizeof(int) * 66);
  cudaMalloc(&d_warpszs, sizeof(int) * 66);
  cudaMalloc(&d_warpids, sizeof(int) * 66);
  builtin<<<2, 33>>>(d_ec, d_laneids, d_warpszs, d_warpids);
  wait_and_check("builtin");
  int laneids[66] = {0}, warpids[66] = {0}, warpszs[66] = {0};
  cudaMemcpy(laneids, d_laneids, sizeof(int) * 66, cudaMemcpyDeviceToHost);
  cudaMemcpy(warpids, d_warpids, sizeof(int) * 66, cudaMemcpyDeviceToHost);
  cudaMemcpy(warpszs, d_warpszs, sizeof(int) * 66, cudaMemcpyDeviceToHost);
  std::map<int, int> cnt_laneid, cnt_warpid, cnt_warpsz, cnt_laneid_num;
  for (int I = 0; I < 66; ++I) {
    cnt_warpid[warpids[I]]++;
    cnt_warpsz[warpszs[I]]++;
    cnt_laneid[laneids[I]]++;
  }

  int total_warpid = 0;
  for (const auto &[k, v] : cnt_warpid)
    total_warpid += v;
  for (const auto &[k, v] : cnt_laneid)
    cnt_laneid_num[v]++;

  auto check_laneid_num = [&]() {
    if (cnt_laneid_num.size() != 2)
      return false;
    const auto first = *cnt_laneid_num.begin();
    const auto second = *std::next(cnt_laneid_num.begin());
    return first.first + 2 == second.first;
  };

  cudaMemset(d_ec, !check_laneid_num(), sizeof(int));
  wait_and_check("builtin");

  cudaMemset(d_ec, total_warpid != 66, sizeof(int));
  wait_and_check("builtin");

  cudaMemset(d_ec, cnt_warpsz.size() > 2U, sizeof(int));
  wait_and_check("builtin");
  cudaMemset(d_ec, 0, sizeof(int));
  cudaFree(d_warpids);
  cudaFree(d_laneids);
  cudaFree(d_warpszs);

  setp<<<1, 1>>>(d_ec);
  wait_and_check("setp");

  lop3<<<1, 1>>>(d_ec);
  wait_and_check("lop3");

  lop3_template<<<1, 1>>>(d_ec);
  wait_and_check("lop3_template");

  cudaFree(d_ec);

  return ret;
}
