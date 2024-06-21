// ====------ math-intrinsics.cu--------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cassert>
#include <iostream>
#include <cuda.h>

__global__ void testMathFunctions(int *TestResults) {

  // int __dp2a_lo(int srcA, int srcB, int c)
  {
    int ret = __dp2a_lo(930681129, 370772529, 2010968336);
    TestResults[0] = ret == 2009507875;
  }

  // unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c)
  {
    unsigned ret = __dp2a_lo(261879580u, 462533001u, 1244651601u);
    TestResults[1] = ret == 1254025336u;
  }

  // int __dp2a_hi(int srcA, int srcB, int c)
  {
    int ret = __dp2a_hi(2033148131, 1987852344, 1836738289);
    TestResults[2] = ret == 1843474575;
  }

  // unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c)
  {
    unsigned ret = __dp2a_hi(3407045239u, 1034879260u, 1566081712u);
    TestResults[3] = ret == 1573664144u;
  }

  // int __dp4a(int srcA, int srcB, int c)
  {
    int ret = __dp4a(-1190208646, 231822748, 1361188354);
    TestResults[4] = ret == 1361171428;
  }

  // unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c)
  {
    unsigned ret = __dp4a(3065883002u, 1618319527u, 3160878852u);
    TestResults[5] = ret == 3160964499u;
  }
}

int main() {
  constexpr int NumberOfTests = 6;
  int *TestResults;
  cudaMallocManaged(&TestResults, NumberOfTests * sizeof(int));
  testMathFunctions<<<1, 1>>>(TestResults);
  cudaDeviceSynchronize();
  for (int i = 0; i < NumberOfTests; i++) {
    if (TestResults[i] == 0) {
      std::cout << "Test " << i << " failed" << std::endl;
      return 1;
    }
  }
  return 0;
}
