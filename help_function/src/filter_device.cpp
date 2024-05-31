// ====------  filter_device.cpp ------------------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#define DPCT_HELPER_VERBOSE

#include "dpct/dpct.hpp"

int main() {
  dpct::filter_device({"CPU"});
  dpct::list_devices();
}
