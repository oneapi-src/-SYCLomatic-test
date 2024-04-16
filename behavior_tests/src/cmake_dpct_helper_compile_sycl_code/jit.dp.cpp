// ====----------------------------- jit.dp.cpp---------- -*- C++ -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared.hpp"

extern "C" 
void foo(int *a, int *b, int *c, const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2);

  if (tid<VEC_LENGTH) {
    a[tid] = b[tid] * c[tid] + SEED;
  }
}

extern "C" {
DPCT_EXPORT void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr,
                             unsigned int localMemSize, void **kernelParams,
                             void **extra) {
  // 3 non-default parameters, 0 default parameters
  dpct::args_selector<3, 0, decltype(foo)> selector(kernelParams, extra);
  auto &a = selector.get<0>();
  auto &b = selector.get<1>();
  auto &c = selector.get<2>();
  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    foo(a, b, c, item_ct1);
  });
}
}
