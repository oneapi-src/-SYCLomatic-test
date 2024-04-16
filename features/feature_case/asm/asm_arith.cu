// ====------ asm_arith.cu --------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// Compile flags: -arch=sm_90 --expt-relaxed-constexpr

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>
#include <string>
#include <math.h>

#define CHECK(ID, S, CMP)                                                      \
  {                                                                            \
    S;                                                                         \
    if (!(CMP)) {                                                              \
      return ID;                                                               \
    }                                                                          \
  }

// clang-format off
__device__ int add() {
  int16_t s16, s16x = 1, s16y = 2;
  uint16_t u16, u16x = 1, u16y = 2;
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;
  int64_t s64, s64x = 1, s64y = 2;
  uint64_t u64, u64x = 1, u64y = 2;
  CHECK(1, asm("add.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y))                                   , s16 == 3                                   );
  CHECK(2, asm("add.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y))                                   , u16 == 3                                   );
  CHECK(3, asm("add.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y))                                   , s32 == 3                                   );
  CHECK(4, asm("add.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y))                                   , u32 == 3                                   );
  CHECK(5, asm("add.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y))                                   , s64 == 3                                   );
  CHECK(6, asm("add.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y))                                   , u64 == 3                                   );
  CHECK(7, asm("add.s32.sat %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(std::numeric_limits<int32_t>::max())), s32 == std::numeric_limits<int32_t>::max() );

  return 0;
}

__device__ int sub() {
  int16_t s16, s16x = 5, s16y = 2;
  uint16_t u16, u16x = 5, u16y = 2;
  int32_t s32, s32x = 5, s32y = 2;
  uint32_t u32, u32x = 5, u32y = 2;
  int64_t s64, s64x = 5, s64y = 2;
  uint64_t u64, u64x = 5, u64y = 2;
  CHECK(1, asm("sub.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)), s16 == 3);
  CHECK(2, asm("sub.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == 3);
  CHECK(3, asm("sub.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 3);
  CHECK(4, asm("sub.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 3);
  CHECK(5, asm("sub.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)), s64 == 3);
  CHECK(6, asm("sub.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == 3);
  return 0;
}

__device__ int mul() {
  int16_t s16, s16x = 1, s16y = 2;
  uint16_t u16, u16x = 1, u16y = 2;
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;
  int64_t s64, s64x = 1, s64y = 2;
  uint64_t u64, u64x = 1, u64y = 2;

  // [UNSUPPORRTED] mul.lo && no overflow
  // CHECK(1, asm("mul.lo.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)), s16 == 2);
  // CHECK(2, asm("mul.lo.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == 2);
  // CHECK(3, asm("mul.lo.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 2);
  // CHECK(4, asm("mul.lo.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 2);
  // CHECK(5, asm("mul.lo.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)), s64 == 2);
  // CHECK(6, asm("mul.lo.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == 2);

  // [UNSUPPORRTED] mul.lo && overflow
  // CHECK(7,  asm("mul.lo.s16 %0, %1, %2;" : "=h"(s16) : "h"(std::numeric_limits<int16_t>::max()), "h"((short)2))                           , s16 == -2);  
  // CHECK(8,  asm("mul.lo.u16 %0, %1, %2;" : "=h"(u16) : "h"(std::numeric_limits<int16_t>::max()), "h"(std::numeric_limits<int16_t>::max())), u16 ==  1);
  // CHECK(9,  asm("mul.lo.s32 %0, %1, %2;" : "=r"(s32) : "r"(std::numeric_limits<int32_t>::max()), "r"(2))                                  , s32 == -2); 
  // CHECK(10, asm("mul.lo.u32 %0, %1, %2;" : "=r"(u32) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max())), u32 ==  1);
  // CHECK(11, asm("mul.lo.s64 %0, %1, %2;" : "=l"(s64) : "l"(std::numeric_limits<int64_t>::max()), "l"(2LL))                                , s64 == -2); 
  // CHECK(12, asm("mul.lo.u64 %0, %1, %2;" : "=l"(u64) : "l"(std::numeric_limits<int64_t>::max()), "l"(std::numeric_limits<int64_t>::max())), u64 ==  1);

  // mul.hi && no overflow
  CHECK(13, asm("mul.hi.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)), s16 == 0);
  CHECK(14, asm("mul.hi.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == 0);
  CHECK(15, asm("mul.hi.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 0);
  CHECK(16, asm("mul.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 0);
  CHECK(17, asm("mul.hi.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)), s64 == 0);
  CHECK(18, asm("mul.hi.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == 0);

  // mul.hi && overflow
  CHECK(19, asm("mul.hi.s16 %0, %1, %2;" : "=h"(s16) : "h"(std::numeric_limits<int16_t>::max()), "h"((short)2))                                                                   , s16 == 0                                            );
  CHECK(20, asm("mul.hi.u16 %0, %1, %2;" : "=h"(u16) : "h"(std::numeric_limits<int16_t>::max()), "h"(std::numeric_limits<int16_t>::max()))                                        , u16 == (std::numeric_limits<int16_t>::max() - 1) / 2);
  CHECK(21, asm("mul.hi.s32 %0, %1, %2;" : "=r"(s32) : "r"(std::numeric_limits<int32_t>::max()), "r"(2))                                                                          , s32 == 0                                            );
  CHECK(22, asm("mul.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max()))                                        , u32 == (std::numeric_limits<int32_t>::max() - 1) / 2);
  CHECK(23, asm("mul.hi.s64 %0, %1, %2;" : "=l"(s64) : "l"(std::numeric_limits<int64_t>::max()), "l"(2LL))                                                                        , s64 == 0                                            );
  CHECK(24, asm("mul.hi.u64 %0, %1, %2;" : "=l"(u64) : "l"((unsigned long long)std::numeric_limits<int64_t>::max()), "l"((unsigned long long)std::numeric_limits<int64_t>::max())), u64 == (std::numeric_limits<int64_t>::max() - 1) / 2);
  
  // mul.wide
  CHECK(25, asm("mul.wide.s16 %0, %1, %2;" : "=r"(s32) : "h"(std::numeric_limits<int16_t>::max()), "h"((short)2))                                   , s32 == std::numeric_limits<int16_t>::max() * 2                                                        );
  CHECK(26, asm("mul.wide.u16 %0, %1, %2;" : "=r"(u32) : "h"(std::numeric_limits<int16_t>::max()), "h"(std::numeric_limits<int16_t>::max()))        , u32 == std::numeric_limits<int16_t>::max() * std::numeric_limits<int16_t>::max()                      );
  CHECK(27, asm("mul.wide.s32 %0, %1, %2;" : "=l"(s64) : "r"(std::numeric_limits<int32_t>::max()), "r"(2))                                          , s64 == (int64_t)std::numeric_limits<int32_t>::max() * (int64_t)2                                      );
  CHECK(28, asm("mul.wide.u32 %0, %1, %2;" : "=l"(u64) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max()))        , u64 ==  (uint64_t)std::numeric_limits<int32_t>::max() *  (uint64_t)std::numeric_limits<int32_t>::max());
  
  return 0;
}

__device__ int mad() {
  int16_t s16, s16x = 1, s16y = 2;
  uint16_t u16, u16x = 1, u16y = 2;
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;
  int64_t s64, s64x = 1, s64y = 2;
  uint64_t u64, u64x = 1, u64y = 2;

  // [UNSUPPORRTED] mad.lo && no overflow
  // CHECK(1, asm("mad.lo.s16 %0, %1, %2, %3;" : "=h"(s16) : "h"(s16x), "h"(s16y), "h"(s16x)), s16 == 3);
  // CHECK(2, asm("mad.lo.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16x), "h"(u16y), "h"(u16x)), u16 == 3);
  // CHECK(3, asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(s32x), "r"(s32y), "r"(s32x)), s32 == 3);
  // CHECK(4, asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32x), "r"(u32y), "r"(u32x)), u32 == 3);
  // CHECK(5, asm("mad.lo.s64 %0, %1, %2, %3;" : "=l"(s64) : "l"(s64x), "l"(s64y), "l"(s64x)), s64 == 3);
  // CHECK(6, asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64x), "l"(u64y), "l"(u64x)), u64 == 3);

  // [UNSUPPORRTED] mad.lo && overflow
  // CHECK(7,  asm("mad.lo.s16 %0, %1, %2, %3;" : "=h"(s16) : "h"((short)std::numeric_limits<int16_t>::max()), "h"((short)2), "h"(s16x))                                                               , s16 == -1);
  // CHECK(8,  asm("mad.lo.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"((unsigned short)std::numeric_limits<int16_t>::max()), "h"((unsigned short)std::numeric_limits<int16_t>::max()), "h"(u16x))           , u16 ==  2);
  // CHECK(9,  asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(std::numeric_limits<int32_t>::max()), "r"(2), "r"(s32x))                                                                             , s32 == -1);
  // CHECK(10, asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max()), "r"(u32x))                                           , u32 ==  2);
  // CHECK(11, asm("mad.lo.s64 %0, %1, %2, %3;" : "=l"(s64) : "l"((long long)std::numeric_limits<int64_t>::max()), "l"((long long)2), "l"(s64x))                                                       , s64 == -1);
  // CHECK(12, asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"((unsigned long long)std::numeric_limits<int64_t>::max()), "l"((unsigned long long)std::numeric_limits<int64_t>::max()), "l"(u64x))   , u64 ==  2);

  // mad.hi && no overflow
  CHECK(13, asm("mad.hi.s16 %0, %1, %2, %3;" : "=h"(s16) : "h"(s16x), "h"(s16y), "h"(s16x)), s16 == 1);
  CHECK(14, asm("mad.hi.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16x), "h"(u16y), "h"(u16x)), u16 == 1);
  CHECK(15, asm("mad.hi.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(s32x), "r"(s32y), "r"(s32x)), s32 == 1);
  CHECK(16, asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32x), "r"(u32y), "r"(u32x)), u32 == 1);
  CHECK(17, asm("mad.hi.s64 %0, %1, %2, %3;" : "=l"(s64) : "l"(s64x), "l"(s64y), "l"(s64x)), s64 == 1);
  CHECK(18, asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64x), "l"(u64y), "l"(u64x)), u64 == 1);

  // mad.hi && overflow
  CHECK(19, asm("mad.hi.s16 %0, %1, %2, %3;" : "=h"(s16) : "h"((short)std::numeric_limits<int16_t>::max()), "h"((short)2), "h"(s16x))                                                             , s16 == 1                  );
  CHECK(20, asm("mad.hi.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"((unsigned short)std::numeric_limits<int16_t>::max()), "h"((unsigned short)std::numeric_limits<int16_t>::max()), "h"(u16x))         , u16 == 16384              );
  CHECK(21, asm("mad.hi.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(std::numeric_limits<int32_t>::max()), "r"(2), "r"(s32x))                                                                           , s32 == 1                  );
  CHECK(22, asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max()), "r"(u32x))                                         , u32 == 1073741824         );
  CHECK(23, asm("mad.hi.s64 %0, %1, %2, %3;" : "=l"(s64) : "l"((long long)std::numeric_limits<int64_t>::max()), "l"((long long)2), "l"(s64x))                                                     , s64 == 1                  );
  CHECK(24, asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"((unsigned long long)std::numeric_limits<int64_t>::max()), "l"((unsigned long long)std::numeric_limits<int64_t>::max()), "l"(u64x)) , u64 == 4611686018427387904);

  // mad.wide
  CHECK(25, asm("mad.wide.s16 %0, %1, %2, %3;" : "=r"(s32) : "h"((short)std::numeric_limits<int16_t>::max()), "h"((short)2), "r"((int)s16x))                                                          , s32 == std::numeric_limits<int16_t>::max() * 2 + 1                                             );
  CHECK(26, asm("mad.wide.u16 %0, %1, %2, %3;" : "=r"(u32) : "h"((unsigned short)std::numeric_limits<int16_t>::max()), "h"((unsigned short)std::numeric_limits<int16_t>::max()), "r"((unsigned)u16x)) , u32 == std::numeric_limits<int16_t>::max() * std::numeric_limits<int16_t>::max() + 1           );
  CHECK(27, asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(s64) : "r"(std::numeric_limits<int32_t>::max()), "r"(2), "l"((long long)s32x))                                                                  , s64 == (int64_t)std::numeric_limits<int32_t>::max() * 2 + 1                                    );
  CHECK(28, asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(u64) : "r"(std::numeric_limits<int32_t>::max()), "r"(std::numeric_limits<int32_t>::max()), "l"((unsigned long long)u32x))                       , u64 ==  (uint64_t)std::numeric_limits<int32_t>::max() * std::numeric_limits<int32_t>::max() + 1);
  
  return 0;
}

__device__ int mul24() {
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;

  CHECK(1, asm("mul24.lo.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 2);
  CHECK(2, asm("mul24.lo.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 2);
 
  // mul24.hi not supported
  return 0;
}

__device__ int mad24() {
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;

  CHECK(1, asm("mad24.lo.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(s32x), "r"(s32y), "r"(s32x)), s32 == 3);
  CHECK(2, asm("mad24.lo.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32x), "r"(u32y), "r"(u32x)), u32 == 3);

  // mad24.hi not supported
  return 0;
}

__device__ int div() {
  int16_t s16, s16x = 4, s16y = 2;
  uint16_t u16, u16x = 4, u16y = 2;
  int32_t s32, s32x = 4, s32y = 2;
  uint32_t u32, u32x = 4, u32y = 2;
  int64_t s64, s64x = 4, s64y = 2;
  uint64_t u64, u64x = 4, u64y = 2;

  CHECK(1, asm("div.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)), s16 == 2);
  CHECK(2, asm("div.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == 2);
  CHECK(3, asm("div.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 2);
  CHECK(4, asm("div.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 2);
  CHECK(5, asm("div.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)), s64 == 2);
  CHECK(6, asm("div.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == 2);

  return 0;
}

__device__ int rem() {
  int16_t s16, s16x = 5, s16y = 2;
  uint16_t u16, u16x = 5, u16y = 2;
  int32_t s32, s32x = 5, s32y = 2;
  uint32_t u32, u32x = 5, u32y = 2;
  int64_t s64, s64x = 5, s64y = 2;
  uint64_t u64, u64x = 5, u64y = 2;

  CHECK(1, asm("rem.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)), s16 == 1);
  CHECK(2, asm("rem.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == 1);
  CHECK(3, asm("rem.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)), s32 == 1);
  CHECK(4, asm("rem.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == 1);
  CHECK(5, asm("rem.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)), s64 == 1);
  CHECK(6, asm("rem.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == 1);

  return 0;
}

__device__ int abs() {
  int16_t s16;
  int32_t s32;
  int64_t s64;
  CHECK(1, asm("abs.s16 %0, %1;" : "=h"(s16) : "h"((int16_t)999))                                , s16 == 999                                          );
  CHECK(2, asm("abs.s32 %0, %1;" : "=r"(s32) : "r"((int32_t)std::numeric_limits<int16_t>::min())), s32 == -(int32_t)std::numeric_limits<int16_t>::min());
  CHECK(3, asm("abs.s64 %0, %1;" : "=l"(s64) : "l"((int64_t)std::numeric_limits<int32_t>::min())), s64 == -(int64_t)std::numeric_limits<int32_t>::min());

  return 0;
}

__device__ int neg() {
  int16_t s16;
  int32_t s32;
  int64_t s64;
  CHECK(1, asm("neg.s16 %0, %1;" : "=h"(s16) : "h"((int16_t)999))                                , s16 == -999                                         );
  CHECK(2, asm("neg.s32 %0, %1;" : "=r"(s32) : "r"((int32_t)std::numeric_limits<int16_t>::min())), s32 == -(int32_t)std::numeric_limits<int16_t>::min());
  CHECK(3, asm("neg.s64 %0, %1;" : "=l"(s64) : "l"((int64_t)std::numeric_limits<int32_t>::min())), s64 == -(int64_t)std::numeric_limits<int32_t>::min());
  return 0;
}

__device__ int min() {
  int16_t s16, s16x = 1, s16y = 2;
  uint16_t u16, u16x = 1, u16y = 2;
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;
  int64_t s64, s64x = 1, s64y = 2;
  uint64_t u64, u64x = 1, u64y = 2;
  CHECK(1, asm("min.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)) , s16 == 1);
  CHECK(2, asm("min.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)) , u16 == 1);
  CHECK(3, asm("min.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)) , s32 == 1);
  CHECK(4, asm("min.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)) , u32 == 1);
  CHECK(5, asm("min.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)) , s64 == 1);
  CHECK(6, asm("min.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)) , u64 == 1);
  CHECK(7, asm("min.relu.s32 %0, %1, %2;" : "=r"(s32) : "r"(-2), "r"(-1)), s32 == 0);
  return 0;
}

__device__ int max() {
  int16_t s16, s16x = 1, s16y = 2;
  uint16_t u16, u16x = 1, u16y = 2;
  int32_t s32, s32x = 1, s32y = 2;
  uint32_t u32, u32x = 1, u32y = 2;
  int64_t s64, s64x = 1, s64y = 2;
  uint64_t u64, u64x = 1, u64y = 2;
  CHECK(1, asm("max.s16 %0, %1, %2;" : "=h"(s16) : "h"(s16x), "h"(s16y)) , s16 == 2);
  CHECK(2, asm("max.u16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)) , u16 == 2);
  CHECK(3, asm("max.s32 %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(s32y)) , s32 == 2);
  CHECK(4, asm("max.u32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)) , u32 == 2);
  CHECK(5, asm("max.s64 %0, %1, %2;" : "=l"(s64) : "l"(s64x), "l"(s64y)) , s64 == 2);
  CHECK(6, asm("max.u64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)) , u64 == 2);
  CHECK(7, asm("max.relu.s32 %0, %1, %2;" : "=r"(s32) : "r"(-2), "r"(-1)), s32 == 0);
  return 0;
}

__device__ int popc() {
  uint32_t res;
  CHECK(1, asm("popc.b32 %0, %1;" : "=r"(res) : "r"(0xFFFF00FFU))          , res == 24);
  CHECK(2, asm("popc.b64 %0, %1;" : "=r"(res) : "l"(0xFF00FF00FF00FF00ULL)), res == 32);
  return 0;
}

__device__ int clz() {
  uint32_t res;
  CHECK(1, asm("clz.b32 %0, %1;" : "=r"(res) : "r"(0x0FFF0000U))           , res ==  4);
  CHECK(2, asm("clz.b64 %0, %1;" : "=r"(res) : "l"(0x00000000FFFFFFFFULL)) , res == 32);
  return 0;
}

__device__ int brev() {
  uint32_t res;
  uint64_t r64;
  CHECK(1, asm("brev.b32 %0, %1;" : "=r"(res) : "r"(0x80000000U)), res == 1);
  CHECK(2, asm("brev.b64 %0, %1;" : "=l"(r64) : "l"(0x8000000000000000ULL)), r64 == 1);
  return 0;
}

__device__ int bitwise_and() {
  uint16_t u16, u16x = 5, u16y = 2;
  uint32_t u32, u32x = 5, u32y = 2;
  uint64_t u64, u64x = 5, u64y = 2;

  CHECK(1, asm("and.b16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == (5 & 2));
  CHECK(2, asm("and.b32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == (5 & 2));
  CHECK(3, asm("and.b64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == (5 & 2));
  return 0;
}

__device__ int bitwise_or() {
  uint16_t u16, u16x = 5, u16y = 2;
  uint32_t u32, u32x = 5, u32y = 2;
  uint64_t u64, u64x = 5, u64y = 2;

  CHECK(1, asm("or.b16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == (5 | 2));
  CHECK(2, asm("or.b32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == (5 | 2));
  CHECK(3, asm("or.b64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == (5 | 2));
  return 0;
}

__device__ int bitwise_xor() {
  uint16_t u16, u16x = 5, u16y = 2;
  uint32_t u32, u32x = 5, u32y = 2;
  uint64_t u64, u64x = 5, u64y = 2;

  CHECK(1, asm("xor.b16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "h"(u16y)), u16 == (5 ^ 2));
  CHECK(2, asm("xor.b32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(u32y)), u32 == (5 ^ 2));
  CHECK(3, asm("xor.b64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "l"(u64y)), u64 == (5 ^ 2));
  return 0;
}

__device__ int bitwise_not() {
  uint16_t u16, u16x = 5;
  uint32_t u32, u32x = 5;
  uint64_t u64, u64x = 5;

  CHECK(1, asm("not.b16 %0, %1;" : "=h"(u16) : "h"(u16x)), u16 == (uint16_t)(~5));
  CHECK(2, asm("not.b32 %0, %1;" : "=r"(u32) : "r"(u32x)), u32 == (uint32_t)(~5));
  CHECK(3, asm("not.b64 %0, %1;" : "=l"(u64) : "l"(u64x)), u64 == (uint64_t)(~5));
  return 0;
}

__device__ int cnot() {
  uint16_t u16, u16x = 0;
  uint32_t u32, u32x = 5;
  uint64_t u64, u64x = 0;

  CHECK(1, asm("cnot.b16 %0, %1;" : "=h"(u16) : "h"(u16x)), u16 == 1);
  CHECK(2, asm("cnot.b32 %0, %1;" : "=r"(u32) : "r"(u32x)), u32 == 0);
  CHECK(3, asm("cnot.b64 %0, %1;" : "=l"(u64) : "l"(u64x)), u64 == 1);
  return 0;
}

__device__ int shl() {
  uint16_t u16, u16x = 5; unsigned x = 2;
  uint32_t u32, u32x = 8; unsigned y = 9;
  uint64_t u64, u64x = 4; unsigned z = 7;

  CHECK(1, asm("shl.b16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "r"(x)), u16 == (5 << 2));
  CHECK(2, asm("shl.b32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(y)), u32 == (8 << 9));
  CHECK(3, asm("shl.b64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "r"(z)), u64 == (4 << 7));
  return 0;
}

__device__ int shr() {
  uint16_t u16, u16x = 1; unsigned x = 4;
  uint32_t u32, u32x = 5; unsigned y = 2;
  uint64_t u64, u64x = 9; unsigned z = 7;

  CHECK(1, asm("shr.b16 %0, %1, %2;" : "=h"(u16) : "h"(u16x), "r"(x)), u16 == (1 >> 4));
  CHECK(2, asm("shr.b32 %0, %1, %2;" : "=r"(u32) : "r"(u32x), "r"(y)), u32 == (5 >> 2));
  CHECK(3, asm("shr.b64 %0, %1, %2;" : "=l"(u64) : "l"(u64x), "r"(z)), u64 == (9 >> 7));
  return 0;
}

template <typename T>
__device__ T deg2rad(T val) {
  constexpr auto PI = 3.14159265358979323846f;
  return val * PI / 180.0f;
}

#define FLOAT_CMP(X, Y) ((X - Y) < 1e-4)
#define POWF2(X) (pow(2.0f, X))

__device__ int asm_copysign() {
  float f32 = 0.0f;
  double f64 = 0.0;
  CHECK(1, asm("copysign.f32 %0, %1, %2;" : "=f"(f32) : "f"(-10.0f), "f"(100.0f)), FLOAT_CMP(f32, -100.0f));
  CHECK(2, asm("copysign.f64 %0, %1, %2;" : "=d"(f64) : "d"(-10.0), "d"(100.0)), FLOAT_CMP(f64, -100.0));
  return 0;
}

__device__ int asm_cos() {
  float f32 = 0.0f;
  CHECK(1, asm("cos.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(90.0f))), FLOAT_CMP(f32, 0.0f));
  CHECK(2, asm("cos.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(0.0f))), FLOAT_CMP(f32, 1.0f));
  CHECK(3, asm("cos.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(60.0f))), FLOAT_CMP(f32, 0.5f));
  CHECK(4, asm("cos.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(180.0f))), FLOAT_CMP(f32, -1.0f));
  CHECK(5, asm("cos.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(90.0f))), FLOAT_CMP(f32, 0.0f));
  CHECK(6, asm("cos.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(0.0f))), FLOAT_CMP(f32, 1.0f));
  CHECK(7, asm("cos.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(60.0f))), FLOAT_CMP(f32, 0.5f));
  CHECK(8, asm("cos.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(180.0f))), FLOAT_CMP(f32, -1.0f));
  return 0;
}

__device__ int asm_sin() {
  float f32 = 0.0f;
  CHECK(1, asm("sin.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(90.0f))), FLOAT_CMP(f32, 1.0f));
  CHECK(2, asm("sin.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(0.0f))), FLOAT_CMP(f32, 0.0f));
  CHECK(3, asm("sin.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(30.0f))), FLOAT_CMP(f32, 0.5f));
  CHECK(4, asm("sin.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(180.0f))), FLOAT_CMP(f32, 0.0f));
  CHECK(5, asm("sin.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(90.0f))), FLOAT_CMP(f32, 1.0f));
  CHECK(6, asm("sin.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(0.0f))), FLOAT_CMP(f32, 0.0f));
  CHECK(7, asm("sin.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(30.0f))), FLOAT_CMP(f32, 0.5f));
  CHECK(8, asm("sin.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(deg2rad(180.0f))), FLOAT_CMP(f32, 0.0f));
  return 0;
}

__device__ int asm_tanh() {
  float f32 = 0.0f;
  CHECK(1, asm("tanh.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(45.0f))), FLOAT_CMP(f32, tanh(deg2rad(45.0f))));
  CHECK(2, asm("tanh.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(0.0f))), FLOAT_CMP(f32, tanh(deg2rad(0.0f))));
  CHECK(3, asm("tanh.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(180.0f))), FLOAT_CMP(f32, tanh(deg2rad(180.0f))));
  CHECK(4, asm("tanh.approx.f32 %0, %1;" : "=f"(f32) : "f"(deg2rad(90.0f))), FLOAT_CMP(f32, tanh(deg2rad(90.0f))));
  return 0;
}

__device__ int asm_ex2() {
  float f32 = 0.0f;
  CHECK(1, asm("ex2.approx.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, POWF2(2.1f)));
  CHECK(2, asm("ex2.approx.f32 %0, %1;" : "=f"(f32) : "f"(3.4f)), FLOAT_CMP(f32, POWF2(3.4f)));
  CHECK(3, asm("ex2.approx.f32 %0, %1;" : "=f"(f32) : "f"(9.7f)), FLOAT_CMP(f32, POWF2(9.7f)));
  CHECK(4, asm("ex2.approx.f32 %0, %1;" : "=f"(f32) : "f"(6.4f)), FLOAT_CMP(f32, POWF2(6.4f)));
  CHECK(5, asm("ex2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, POWF2(2.1f)));
  CHECK(6, asm("ex2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(3.4f)), FLOAT_CMP(f32, POWF2(3.4f)));
  CHECK(7, asm("ex2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(9.7f)), FLOAT_CMP(f32, POWF2(9.7f)));
  CHECK(8, asm("ex2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(6.4f)), FLOAT_CMP(f32, POWF2(6.4f)));
  return 0;
}

__device__ int asm_lg2() {
  float f32 = 0.0f;
  CHECK(1, asm("lg2.approx.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, log2(2.1f)));
  CHECK(2, asm("lg2.approx.f32 %0, %1;" : "=f"(f32) : "f"(3.4f)), FLOAT_CMP(f32, log2(3.4f)));
  CHECK(3, asm("lg2.approx.f32 %0, %1;" : "=f"(f32) : "f"(9.7f)), FLOAT_CMP(f32, log2(9.7f)));
  CHECK(4, asm("lg2.approx.f32 %0, %1;" : "=f"(f32) : "f"(6.4f)), FLOAT_CMP(f32, log2(6.4f)));
  CHECK(5, asm("lg2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, log2(2.1f)));
  CHECK(6, asm("lg2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(3.4f)), FLOAT_CMP(f32, log2(3.4f)));
  CHECK(7, asm("lg2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(9.7f)), FLOAT_CMP(f32, log2(9.7f)));
  CHECK(8, asm("lg2.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(6.4f)), FLOAT_CMP(f32, log2(6.4f)));
  return 0;
}

__device__ int sad() {
  int16_t s16;
  uint16_t u16;
  int32_t s32;
  uint32_t u32;
  int64_t s64;
  uint64_t u64;
  CHECK(1, asm("sad.s16 %0, %1, %2, %3;" : "=h"(s16) : "h"((int16_t)-1), "h"((int16_t)3), "h"((int16_t)5)), s16 == 9);
  CHECK(2, asm("sad.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"((int16_t)1), "h"((int16_t)3), "h"((int16_t)5)), u16 == 7);
  CHECK(3, asm("sad.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(-1), "r"(3), "r"(5)), s32 == 9);
  CHECK(4, asm("sad.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(1), "r"(3), "r"(5)), u32 == 7);
  CHECK(5, asm("sad.s64 %0, %1, %2, %3;" : "=l"(s64) : "l"(-1ll), "l"(3ll), "l"(5ll)), s64 == 9);
  CHECK(6, asm("sad.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(1ll), "l"(3ll), "l"(5ll)), u64 == 7);
  return 0;
}

__device__ int asm_rsqrt() {
  float f32;
  double f64;
  CHECK(1, asm("rsqrt.approx.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, rsqrt(2.1f)));
  CHECK(2, asm("rsqrt.approx.f64 %0, %1;" : "=d"(f64) : "d"(2.1)), FLOAT_CMP(f64, rsqrt(2.1)));
  return 0;
}

__device__ int asm_sqrt() {
  float f32;
  double f64;
  CHECK(1, asm("sqrt.approx.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(2, asm("sqrt.approx.f32.ftz %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(3, asm("sqrt.rn.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(4, asm("sqrt.rz.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(5, asm("sqrt.rm.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(6, asm("sqrt.rp.f32 %0, %1;" : "=f"(f32) : "f"(2.1f)), FLOAT_CMP(f32, sqrt(2.1f)));
  CHECK(7, asm("sqrt.rn.f64 %0, %1;" : "=d"(f64) : "d"(2.1)), FLOAT_CMP(f64, sqrt(2.1)));
  CHECK(8, asm("sqrt.rz.f64 %0, %1;" : "=d"(f64) : "d"(2.1)), FLOAT_CMP(f64, sqrt(2.1)));
  CHECK(9, asm("sqrt.rm.f64 %0, %1;" : "=d"(f64) : "d"(2.1)), FLOAT_CMP(f64, sqrt(2.1)));
  CHECK(10, asm("sqrt.rp.f64 %0, %1;" : "=d"(f64) : "d"(2.1)), FLOAT_CMP(f64, sqrt(2.1)));
  return 0;
}

__device__ int testp() {
  int pred = 0;
  { asm(".reg .pred p1; testp.finite.f32 p1, %1;  @p1 mov.s32 %0, 1;" : "=r"(pred) : "f"(0.1f)); if (!pred) { return 1; } };
  { asm(".reg .pred p2; testp.infinite.f32 p2, %1; @p2 mov.s32 %0, 1;" : "=r"(pred) : "f"(std::numeric_limits<float>::infinity())); if (!pred) { return 2; } };
  { asm(".reg .pred p3; testp.number.f32 p3, %1; @p3 mov.s32 %0, 1;" : "=r"(pred) : "f"(9.7f)); if (!pred) { return 3; } };
  { asm(".reg .pred p4; testp.notanumber.f32 p4, %1; @p4 mov.s32 %0, 1;" : "=r"(pred) : "f"(NAN)); if (!pred) { return 4; } };
  { asm(".reg .pred p5; testp.normal.f32 p5, %1; @p5 mov.s32 %0, 1;" : "=r"(pred) : "f"(9.5f)); if (!pred) { return 5; } };
  { asm(".reg .pred p6; testp.subnormal.f32 p6, %1; @p6 mov.s32 %0, 1;" : "=r"(pred) : "f"(0.1e-300f)); if (!pred) { return 6; } };
  { asm(".reg .pred p7; testp.finite.f64 p7, %1; @p7 mov.s32 %0, 1;" : "=r"(pred) : "d"(0.1)); if (!pred) { return 1; } };
  { asm(".reg .pred p8; testp.infinite.f64 p8, %1; @p8 mov.s32 %0, 1;" : "=r"(pred) : "d"(std::numeric_limits<double>::infinity())); if (!pred) { return 2; } };
  { asm(".reg .pred p9; testp.number.f64 p9, %1; @p9 mov.s32 %0, 1;" : "=r"(pred) : "d"(9.7)); if (!pred) { return 3; } };
  { asm(".reg .pred p10; testp.notanumber.f64 p10, %1; @p10 mov.s32 %0, 1;" : "=r"(pred) : "d"(double(NAN))); if (!pred) { return 4; } };
  { asm(".reg .pred p11; testp.normal.f64 p11, %1; @p11 mov.s32 %0, 1;" : "=r"(pred) : "d"(9.5)); if (!pred) { return 5; } };
  { asm(".reg .pred p12; testp.subnormal.f64 p12, %1; @p12 mov.s32 %0, 1;" : "=r"(pred) : "d"(0.1e-400)); if (!pred) { return 6; } };
  return 0;
}

__device__ int dp2a() {
  int32_t i32;
  uint32_t u32;
  { asm("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(930681129), "r"(370772529), "r"(2010968336)); if (!(i32 == 2009507875)) { return 1; } };
  { asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(i32) : "r"(-1784870143), "r"(3550903701u), "r"(929114859)); if (!(i32 == 926130217)) { return 2; } };
  { asm("dp2a.lo.u32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(3526794897u), "r"(1440743042), "r"(370074542)); if (!(i32 == 364852196)) { return 3; } };
  { asm("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(261879580u), "r"(462533001u), "r"(1244651601u)); if (!(u32 == 1254025336u)) { return 4; } };
  { asm("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(2033148131), "r"(1987852344), "r"(1836738289)); if (!(i32 == 1843474575)) { return 5; } };
  { asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(i32) : "r"(925779231), "r"(2297216285u), "r"(-2134129287)); if (!(i32 == -2128032131)) { return 6; } };
  { asm("dp2a.hi.u32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(1465064346u), "r"(-987065627), "r"(511196861)); if (!(i32 == 510174688)) { return 7; } };
  { asm("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(3407045239u), "r"(1034879260u), "r"(1566081712u)); if (!(u32 == 1573664144u)) { return 8; } };
  return 0;
}

__device__ int dp4a() {
  int32_t i32;
  uint32_t u32;
  { asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(-1190208646), "r"(231822748), "r"(1361188354)); if (!(i32 == 1361171428)) { return 1; } };
  { asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(i32) : "r"(851192907), "r"(4159889898u), "r"(-1560201465)); if (!(i32 == -1560178121)) { return 2; } };
  { asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(908604347u), "r"(1279608234), "r"(-1450969803)); if (!(i32 == -1450975502)) { return 3; } };
  { asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(3065883002u), "r"(1618319527u), "r"(3160878852u)); if (!(u32 == 3160964499u)) { return 4; } };
  return 0;
}

__device__ int bfe() {
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  { asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(0x00FFFF00), "r"(8), "r"(16)); if (i32 != -1) { return 1; } };
  { asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(0xFF000000), "r"(24), "r"(16)); if (i32 != -1) { return 2; } };
  { asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(0x0F000000), "r"(24), "r"(16)); if (i32 != 15) { return 3; } };
  { asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(0xFF000000), "r"(32), "r"(16)); if (i32 != -1) { return 4; } };
  { asm("bfe.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(0x0F000000), "r"(32), "r"(16)); if (i32 != 0) { return 5; } };
  { asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(0x000000FF), "r"(0), "r"(9)); if (u32 != 255) { return 6; } };
  { asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(0x00FFFF00), "r"(8), "r"(16)); if (u32 != 65535) { return 7; } };
  { asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(0xFF000000), "r"(24), "r"(16)); if (u32 != 255) { return 8; } };
  { asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(0xFF000000), "r"(32), "r"(16)); if (u32 != 0) { return 9; } };
  { asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(0x0F000000), "r"(32), "r"(16)); if (u32 != 0) { return 10; } };
  { asm("bfe.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(0x00FFFF00ll), "r"(8), "r"(16)); if (i64 != -1) { return 11; } };
  { asm("bfe.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(0xFF000000ll), "r"(24), "r"(16)); if (i64 != 255) { return 12; } };
  { asm("bfe.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(0x0F000000ll), "r"(24), "r"(16)); if (i64 != 15) { return 13; } };
  { asm("bfe.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(0xFF000000ll), "r"(32), "r"(16)); if (i64 != 0) { return 14; } };
  { asm("bfe.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(0x0F000000ll), "r"(32), "r"(16)); if (i64 != 0) { return 15; } };
  { asm("bfe.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(0x000000FFull), "r"(0), "r"(9)); if (u64 != 255) { return 16; } };
  { asm("bfe.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(0x00FFFF00ull), "r"(8), "r"(16)); if (u64 != 65535) { return 17; } };
  { asm("bfe.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(0xFF000000ull), "r"(24), "r"(16)); if (u64 != 255) { return 18; } };
  { asm("bfe.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(0xFF000000ull), "r"(32), "r"(16)); if (u64 != 0) { return 19; } };
  { asm("bfe.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(0x0F000000ull), "r"(32), "r"(16)); if (u64 != 0) { return 20; } };
  return 0;
}

__device__ int bfi() {
  uint32_t u32;
  uint64_t u64;
  { asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(u32) : "r"(0x0000FFFFu), "r"(0x00FF0000u), "r"(0), "r"(16)); if (u32 != 0x00FFFFFFu) { return 1; } };
  { asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(u32) : "r"(0x000000FFu), "r"(0x00FF0000u), "r"(0), "r"(8)); if (u32 != 0x00FF00FFu) { return 2; } };
  { asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(u64) : "l"(0x0000FFFFull), "l"(0x00FF0000ull), "r"(0), "r"(16)); if (u64 != 0x00FFFFFFull) { return 3; } };
  { asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(u64) : "l"(0x000000FFull), "l"(0x00FF0000ull), "r"(0), "r"(8)); if (u64 != 0x00FF00FFull) { return 4; } };
  return 0;
}
// clang-format on

__global__ void test(int *ec) {
#define TEST(F)                                                                \
  {                                                                            \
    int res = F();                                                             \
    if (res != 0) {                                                            \
      printf("Test " #F " failed\n");                                          \
      *ec = res;                                                               \
      return;                                                                  \
    }                                                                          \
  }

  TEST(add);
  TEST(sub);
  TEST(mul);
  TEST(mad);
  TEST(mul24);
  TEST(mad24);
  TEST(div);
  TEST(rem);
  TEST(abs);
  TEST(neg);
  TEST(min);
  TEST(max);
  TEST(shl);
  TEST(shr);
  TEST(clz);
  TEST(popc);
  TEST(cnot);
  TEST(bitwise_and);
  TEST(bitwise_or);
  TEST(bitwise_not);
  TEST(bitwise_xor);
  TEST(cnot);
  TEST(shl);
  TEST(shr);
  TEST(asm_copysign);
  TEST(asm_cos);
  TEST(asm_sin);
  TEST(asm_tanh);
  TEST(asm_ex2);
  TEST(asm_lg2);
  TEST(sad);
  TEST(asm_rsqrt);
  TEST(asm_sqrt);
  TEST(testp);
  TEST(brev);
  TEST(dp2a);
  TEST(dp4a);
  TEST(bfe);
  TEST(bfi);

  *ec = 0;
}

int main() {
  int *ec;
  cudaMallocManaged(&ec, sizeof(int));
  test<<<1, 1>>>(ec);
  cudaDeviceSynchronize();
  if (*ec != 0) {
    printf("Test asm integer arithmetic instructions failed: %d\n", *ec);
    return 1;
  }
  printf("Test asm integer arithmetic instructions pass.\n");
  return 0;
}
