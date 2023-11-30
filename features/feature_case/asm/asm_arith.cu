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
  TEST(brev);

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
