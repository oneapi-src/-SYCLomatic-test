// ====-------------- shared.hpp-------------------------- -*- C++ -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#define checkErrors(err)  privCheckErrors (err, __FILE__, __LINE__)

inline void privCheckErrors(int result, const char *file, const int line) {
}

#define VEC_LENGTH 128
#define SEED       59
