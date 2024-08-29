// ===------------- math-emu-half.cu------------------- *- CUDA -* --------===//
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

#include "cuda_fp16.h"

using namespace std;

typedef pair<__half, int> hi_pair;

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
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision < 0 ? 0 : precision)
       << Result << " (expect " << Expect - pow(10, -precision) << " ~ "
       << Expect + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<__half> &Inputs,
                 const __half &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__half2float(it));
  }
  float FExpect = __half2float(Expect);
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

// Arithmetic Functions

__global__ void hadd_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hadd_sat(Input1, Input2);
}

void testHadd_sat(float *const Result, __half Input1, __half Input2) {
  hadd_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHadd_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHadd_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hadd_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hfma_sat(float *const Result, __half Input1, __half Input2,
                         __half Input3) {
  *Result = __hfma_sat(Input1, Input2, Input3);
}

void testHfma_sat(float *const Result, __half Input1, __half Input2,
                  __half Input3) {
  hfma_sat<<<1, 1>>>(Result, Input1, Input2, Input3);
  cudaDeviceSynchronize();
}

void testHfma_satCases(const vector<pair<vector<__half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHfma_sat(Result, TestCase.first[0], TestCase.first[1],
                 TestCase.first[2]);
    checkResult("__hfma_sat", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void hmul_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hmul_sat(Input1, Input2);
}

void testHmul_sat(float *const Result, __half Input1, __half Input2) {
  hmul_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHmul_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHmul_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hmul_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hsub_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hsub_sat(Input1, Input2);
}

void testHsub_sat(float *const Result, __half Input1, __half Input2) {
  hsub_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHsub_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHsub_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hsub_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

// Comparison Functions

void checkResult(const string &FuncName, const vector<__half> &Inputs,
                 const bool &Expect, const bool &Result) {
  cout << FuncName << "(" << __half2float(Inputs[0]);
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << __half2float(Inputs[i]);
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

__global__ void hequ(bool *const Result, __half Input1, __half Input2) {
  *Result = __hequ(Input1, Input2);
}

void testHequ(bool *const Result, __half Input1, __half Input2) {
  hequ<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHequCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHequ(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hequ", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgeu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hgeu(Input1, Input2);
}

void testHgeu(bool *const Result, __half Input1, __half Input2) {
  hgeu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgeuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgeu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgeu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgtu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hgtu(Input1, Input2);
}

void testHgtu(bool *const Result, __half Input1, __half Input2) {
  hgtu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgtuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgtu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgtu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hleu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hleu(Input1, Input2);
}

void testHleu(bool *const Result, __half Input1, __half Input2) {
  hleu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHleuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHleu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hleu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hltu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hltu(Input1, Input2);
}

void testHltu(bool *const Result, __half Input1, __half Input2) {
  hltu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHltuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHltu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hltu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hneu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hneu(Input1, Input2);
}

void testHneu(bool *const Result, __half Input1, __half Input2) {
  hneu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHneuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHneu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hneu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

// Half Math Functions

__global__ void _hceil(float *const Result, __half Input1) {
  *Result = hceil(Input1);
}

void testHceilCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hceil<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hceil", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hcos(float *const Result, __half Input1) {
  *Result = hcos(Input1);
}

void testHcosCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hcos<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hcos", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp(float *const Result, __half Input1) {
  *Result = hexp(Input1);
}

void testHexpCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp10(float *const Result, __half Input1) {
  *Result = hexp10(Input1);
}

void testHexp10Cases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp10", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp2(float *const Result, __half Input1) {
  *Result = hexp2(Input1);
}

void testHexp2Cases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp2", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hfloor(float *const Result, __half Input1) {
  *Result = hfloor(Input1);
}

void testHfloorCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hfloor<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hfloor", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog(float *const Result, __half Input1) {
  *Result = hlog(Input1);
}

void testHlogCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog10(float *const Result, __half Input1) {
  *Result = hlog10(Input1);
}

void testHlog10Cases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog10", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog2(float *const Result, __half Input1) {
  *Result = hlog2(Input1);
}

void testHlog2Cases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog2", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hrcp(float *const Result, __half Input1) {
  *Result = hrcp(Input1);
}

void testHrcpCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hrcp<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hrcp", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hrint(float *const Result, __half Input1) {
  *Result = hrint(Input1);
}

void testHrintCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hrint<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hrint", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hrsqrt(float *const Result, __half Input1) {
  *Result = hrsqrt(Input1);
}

void testHrsqrtCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hrsqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hrsqrt", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hsin(float *const Result, __half Input1) {
  *Result = hsin(Input1);
}

void testHsinCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hsin<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hsin", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hsqrt(float *const Result, __half Input1) {
  *Result = hsqrt(Input1);
}

void testHsqrtCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hsqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hsqrt", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _htrunc(float *const Result, __half Input1) {
  *Result = htrunc(Input1);
}

void testHtruncCases(const vector<pair<__half, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _htrunc<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("htrunc", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

int main() {
  testHadd_satCases({
      {{-0.3, -0.4}, {0, 37}},
      {{0.3, -0.4}, {0, 37}},
      {{0.3, 0.4}, {0.7001953125, 16}},
      {{0.3, 0.8}, {1, 15}},
      {{3, 4}, {1, 15}},
  });
  testHfma_satCases({
      {{-0.3, -0.4, -0.2}, {0, 37}},
      {{0.3, -0.4, -0.1}, {0, 37}},
      {{0.3, 0.4, 0.1}, {0.219970703125, 16}},
      {{0.3, 0.4, 0}, {0.1199951171875, 17}},
      {{3, 4, 5}, {1, 15}},
  });
  testHmul_satCases({
      {{-0.3, 0.4}, {0, 37}},
      {{0.3, -4}, {0, 37}},
      {{0.3, 0.4}, {0.1199951171875, 17}},
      {{0.3, 0.8}, {0.239990234375, 16}},
      {{3, 4}, {1, 15}},
  });
  testHsub_satCases({
      {{0, -0.4}, {0.39990234375, 16}},
      {{0.3, -0.4}, {0.7001953125, 16}},
      {{0.3, 0.4}, {0, 37}},
      {{0.3, -0.8}, {1, 15}},
      {{1, 4}, {0, 37}},
  });
  testHequCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, true},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHgeuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, true},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHgtuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, false},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHleuCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, true},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  testHltuCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, false},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  testHneuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, false},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  testHceilCases({
      {-0.3, {0, 37}},
      {0.34, {1, 15}},
      {0.8, {1, 15}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  testHcosCases({
      {-0.3, {0.955, 3}},
      {0.34, {0.943, 3}},
      {0.8, {0.6968, 4}},
      {23, {-0.533, 3}},
      {-12, {0.844, 3}},
  });
  testHexpCases({
      {-0.3, {0.74072265625, 16}},
      {0.34, {1.4052734375, 15}},
      {0.8, {2.224609375, 15}},
      {10, {22032, -2}},
      {-12, {0.00000613927841186523, 20}},
  });
  testHexp10Cases({
      {-0.3, {0.501, 3}},
      {0.34, {2.188, 3}},
      {0.8, {6.31, 2}},
      {4, {10000, 12}},
      {-12, {0, 11}},
  });
  testHexp2Cases({
      {-0.3, {0.812, 3}},
      {0.34, {1.266, 3}},
      {0.8, {1.741, 3}},
      {12, {4096, 12}},
      {-12, {0.000244140625, 19}},
  });
  testHfloorCases({
      {-0.3, {-1, 15}},
      {0.34, {0, 37}},
      {0.8, {0, 37}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  testHlogCases({
      {0.3, {-1.204, 3}},
      {0.34, {-1.078, 3}},
      {0.8, {-0.223, 3}},
      {23, {3.13, 2}},
      {12, {2.48, 2}},
  });
  testHlog10Cases({
      {0.3, {-0.523, 3}},
      // {0.34, {-0.4685, 4}},
      {0.8, {-0.097, 4}},
      {23, {1.361, 3}},
      {12, {1.0791, 4}},
  });
  testHlog2Cases({
      {0.3, {-1.736, 3}},
      {0.34, {-1.556, 3}},
      {0.8, {-0.3223, 4}},
      {23, {4.523, 3}},
      {12, {3.586, 3}},
  });
  testHrcpCases({
      {-0.3, {-3.332, 3}},
      {0.34, {2.939, 3}},
      {0.8, {1.25, 3}},
      {23, {0.0435, 4}},
      {-12, {-0.0833, 4}},
  });
  testHrintCases({
      {-0.3, {0, 37}},
      {0.34, {0., 37}},
      {0.8, {1, 15}},
      {23, {23, 14}},
      {-12, {-12, 14}},
  });
  testHrsqrtCases({
      {0.3, {1.8251953125, 15}},
      {0.34, {1.71484375, 15}},
      {0.8, {1.1181640625, 15}},
      {23, {0.20849609375, 16}},
      {12, {0.28857421875, 16}},
  });
  testHsinCases({
      {-0.3, {-0.2957, 4}},
      {0.34, {0.3335, 4}},
      {0.8, {0.7173, 4}},
      {23, {-0.8462, 4}},
      {-12, {0.5366, 4}},
  });
  testHsqrtCases({
      {0.3, {0.5479, 4}},
      {0.34, {0.583, 3}},
      {0.8, {0.895, 3}},
      {23, {4.8, 2}},
      {12, {3.465, 3}},
  });
  testHtruncCases({
      {-0.3, {0, 37}},
      {0.34, {0, 37}},
      {0.8, {0, 37}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
