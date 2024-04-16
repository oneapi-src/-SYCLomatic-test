// ====---------- math-ext-bf16-conv-double.cu---------- *- CUDA -* -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_bf16.h"

using namespace std;

typedef pair<__nv_bfloat16, int> bf16i_pair;

int passed = 0;
int failed = 0;

void check(bool IsPassed) {
  if (IsPassed) {
    cout << " ---- passed" << endl;
    passed++;
  } else {
    cout << " ---- failed" << endl;
    failed++;
  }
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision < 0 ? 0 : precision)
       << Result << " (expect " << Expect - pow(10, -precision) << " ~ "
       << Expect + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat16 &Expect, const float &Result,
                 const int precision) {
  float FExpect = __bfloat162float(Expect);
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

// Bfloat16 Precision Conversion and Data Movement

__global__ void double2bfloat16(float *const Result, double Input1) {
  *Result = __double2bfloat16(Input1);
}

void testDouble2bfloat16Cases(
    const vector<pair<double, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    double2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__double2bfloat16", {(float)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testDouble2bfloat16Cases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
