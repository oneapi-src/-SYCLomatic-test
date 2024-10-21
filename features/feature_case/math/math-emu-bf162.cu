//====---------- math-emu-bf162.cu ---------- *- CUDA -* ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===----------------------------------------------------------------------===//

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_bf16.h"

using namespace std;

typedef pair<__nv_bfloat162, __nv_bfloat162> bf162_pair;
typedef vector<__nv_bfloat162> bf162_vector;
typedef pair<__nv_bfloat162, int> bf162i_pair;

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

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << fixed << setprecision(precision) << "{" << Result.x << ", "
       << Result.y << "} (expect {" << Expect.x - pow(10, -precision) << " ~ "
       << Expect.x + pow(10, -precision) << ", "
       << Expect.y - pow(10, -precision) << " ~ "
       << Expect.y + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result.x - Expect.x) < pow(10, -precision) &&
        abs(Result.y - Expect.y) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const bool &Expect, const bool &Result) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const __nv_bfloat162 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  float2 FExpect{__bfloat162float(Expect.x), __bfloat162float(Expect.y)};
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const bool &Expect, const bool &Result) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result);
}

// Bfloat162 Arithmetic Functions

__global__ void h2div(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __h2div(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2divCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    h2div<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__h2div", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void habs2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __habs2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHabs2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    habs2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__habs2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hadd2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hadd2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hadd2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hadd2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hadd2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hadd2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hcmadd(float *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hcmadd(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHcmaddCases(const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hcmadd<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                     TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hcmadd", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hfma2(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2Cases(const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma2<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                    TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma2", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hfma2_sat(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2_satCases(
    const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma2_sat<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                        TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma2_sat", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hmul2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hmul2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmul2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hmul2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmul2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hmul2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hneg2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __hneg2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHneg2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hneg2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__hneg2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hsub2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hsub2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hsub2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hsub2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hsub2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hsub2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

// Bfloat162 Comparison Functions

__global__ void hbeq2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hbeq2(Input1, Input2);
}

void testHbeq2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbeq2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbeq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbequ2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbequ2(Input1, Input2);
}

void testHbequ2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbequ2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbequ2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbge2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hbge2(Input1, Input2);
}

void testHbge2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbge2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbge2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbgeu2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbgeu2(Input1, Input2);
}

void testHbgeu2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbgeu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbgt2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hbgt2(Input1, Input2);
}

void testHbgt2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbgt2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbgt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbgtu2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbgtu2(Input1, Input2);
}

void testHbgtu2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbgtu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hble2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hble2(Input1, Input2);
}

void testHble2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hble2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hble2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbleu2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbleu2(Input1, Input2);
}

void testHbleu2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbleu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hblt2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hblt2(Input1, Input2);
}

void testHblt2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hblt2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hblt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbltu2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbltu2(Input1, Input2);
}

void testHbltu2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbltu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbne2(bool *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  *Result = __hbne2(Input1, Input2);
}

void testHbne2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbne2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hbneu2(bool *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2) {
  *Result = __hbneu2(Input1, Input2);
}

void testHbneu2Cases(const vector<pair<bf162_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hbneu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hbneu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void heq2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __heq2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHeq2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    heq2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__heq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hequ2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hequ2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHequ2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hequ2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hequ2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hge2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __hge2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHge2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hge2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hge2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgeu2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hgeu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgeu2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgeu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgt2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __hgt2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgt2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgt2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgtu2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hgtu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgtu2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgtu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hisnan2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __hisnan2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHisnan2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hisnan2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__hisnan2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hle2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __hle2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHle2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hle2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hle2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hleu2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hleu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHleu2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hleu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hlt2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __hlt2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHlt2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hlt2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hlt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hltu2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hltu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHltu2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hltu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmax2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hmax2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmax2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmax2<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmax2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmax2_nan(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hmax2_nan(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmax2_nanCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmax2_nan<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmax2_nan<<<1, 1>>>(Result, {NAN, 1}, {1, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({nan, 1}, {1, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmax2_nan<<<1, 1>>>(Result, {1, NAN}, {NAN, 1});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({1, nan}, {nan, 1}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmax2_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax2_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmin2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hmin2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmin2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmin2<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmin2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmin2_nan(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hmin2_nan(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmin2_nanCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmin2_nan<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmin2_nan<<<1, 1>>>(Result, {NAN, 1}, {1, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({nan, 1}, {1, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmin2_nan<<<1, 1>>>(Result, {1, NAN}, {NAN, 1});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({1, nan}, {nan, 1}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmin2_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin2_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hne2(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 Input2) {
  auto ret = __hne2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHne2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hne2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hneu2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hneu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHneu2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hneu2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hneu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

// Bfloat162 Math Functions

__global__ void _h2ceil(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2ceil(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2ceilCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2ceil<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2ceil", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}
__global__ void _h2cos(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2cos(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2cosCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2cos<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2cos", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2exp(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2exp(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2expCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2exp<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2exp", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2exp10(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2exp10(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2exp10Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2exp10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2exp10", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2exp2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2exp2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2exp2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2exp2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2exp2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2floor(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2floor(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2floorCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2floor<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2floor", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2log(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2log(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2logCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2log<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2log", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2log10(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2log10(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2log10Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2log10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2log10", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2log2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2log2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2log2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2log2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2log2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2rcp(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2rcp(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2rcpCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2rcp<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2rcp", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2rint(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2rint(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2rintCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2rint<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2rint", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2rsqrt(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2rsqrt(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2rsqrtCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2rsqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2rsqrt", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2sin(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2sin(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2sinCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2sin<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2sin", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2sqrt(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2sqrt(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2sqrtCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2sqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2sqrt", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void _h2trunc(float *const Result, __nv_bfloat162 Input1) {
  auto ret = h2trunc(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2truncCases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _h2trunc<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("h2trunc", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

int main() {
  testH2divCases({
      {{{-0.3, -5}, {0.4, 6}}, {{-0.75, -0.83203125}, 16}},
      {{{0.3, 5}, {-4, 0.6}}, {{-0.0751953125, 8.3125}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.75, 2.5}, 15}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.75, 0.83203125}, 16}},
      {{{3, 5}, {4, 6}}, {{0.75, 0.83203125}, 16}},
  });
  testHabs2Cases({
      {{-0.3, -5}, {{0.30078125, 5}, 15}},
      {{0.3, 5}, {{0.30078125, 5}, 15}},
      {{0.4, 0.2}, {{0.400390625, 0.2001953125}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{3, 5}, {{3, 5}, 15}},
  });
  testHadd2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.703125, -1.1015625}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099609375, 1.1015625}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1.1015625}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHadd2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.703125, -1.1015625}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099609375, 1.1015625}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1.1015625}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHadd2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0, 1}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1}, 15}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHcmaddCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{-0.38, -0.32}, 2}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}},
       {{-0.5234375, 0.1806640625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.12060546875, 0.359375}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{-0.18, 0.68}, 2}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{-13, 46}, 14}},
  });
  testHfma2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}},
       {{-0.07958984375, -0.3984375}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{-0.220703125, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.220703125, 0.2001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{0.12060546875, 0.6015625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{17, 38}, 14}},
  });
  testHfma2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{0, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.220703125, 0.2001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{0.12060546875, 0.6015625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{1, 1}, 15}},
  });
  testHmul2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{12, 30}, 14}},
  });
  testHmul2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{12, 30}, 14}},
  });
  testHmul2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHneg2Cases({
      {{-0.3, -5}, {{0.30078125, 5}, 15}},
      {{0.3, 5}, {{-0.30078125, -5}, 15}},
      {{0.4, 0.2}, {{-0.400390625, -0.2001953125}, 16}},
      {{0.3, 0.5}, {{-0.30078125, -0.5}, 16}},
      {{3, 5}, {{-3, -5}, 15}},
  });
  testHsub2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, -0.1015625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{-0.099609375, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{-0.099609375, -0.1015625}, 17}},
      {{{3, 5}, {4, 6}}, {{-1, -1}, 15}},
  });
  testHsub2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, -0.1015625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{-0.099609375, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{-0.099609375, -0.1015625}, 17}},
      {{{3, 5}, {4, 6}}, {{-1, -1}, 15}},
  });
  testHsub2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, 0}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0, 0}, 37}},
      {{{3, 5}, {-4, -6}}, {{1, 1}, 15}},
  });
  testHbeq2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbequ2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHbge2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, true},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbgeu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, true},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHbgt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 2}, {1, 1}}, false},
  });
  testHbgtu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 2}, {1, 1}}, true},
  });
  testHble2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbleu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHblt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, false},
  });
  testHbltu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, true},
  });
  testHbne2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, false},
  });
  testHbneu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, true},
  });
  testHeq2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHequ2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHge2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHgeu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHgt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHgtu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  testHisnan2Cases({
      {{0, 0}, {{0, 0}, 37}},
      {{0.7, 2}, {{0, 0}, 37}},
      {{NAN, 1}, {{1, 0}, 15}},
      {{NAN, NAN}, {{1, 1}, 15}},
      {{0, NAN}, {{0, 1}, 15}},
  });
  testHle2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHleu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHlt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 0}, 37}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHltu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 0}, 37}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  testHmax2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.69921875, 0.69921875}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.69921875, 2}, 15}},
      {{{1, 1}, {4, NAN}}, {{4, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHmax2_nanCases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.69921875, 0.69921875}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.69921875, 2}, 15}},
  });
  testHmin2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{-0.400390625, -0.6015625}, 16}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.400390625, 0.69921875}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.69921875, 2}, 15}},
      {{{1, 1}, {4, NAN}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHmin2_nanCases({
      {{{0, 0}, {-0.4, -0.6}}, {{-0.400390625, -0.6015625}, 16}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.400390625, 0.69921875}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.69921875, 2}, 15}},
  });
  testHne2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHneu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  testH2ceilCases({
      {{-0.3, -0.6}, {{0, 0}, 37}},
      {{0.34, 0.7}, {{1, 1}, 15}},
      {{0.8, 2}, {{1, 2}, 15}},
      {{23, 6}, {{23, 6}, 14}},
      {{-12, 1}, {{-12, 1}, 15}},
  });
  testH2cosCases({
      {{-0.3, -0.6}, {{0.96, 0.82}, 2}},
      {{0.34, 0.7}, {{0.94140625, 0.765625}, 16}},
      {{0.8, 2}, {{0.6953125, -0.416015625}, 16}},
      {{23, 6}, {{-0.53125, 0.9609375}, 16}},
      {{-12, 1}, {{0.84375, 0.5390625}, 16}},
  });
  testH2expCases({
      {{-0.3, -0.6}, {{0.74, 0.55}, 2}},
      {{0.34, 0.7}, {{1.40625, 2.015625}, 15}},
      {{0.8, 2}, {{2.234375, 7.375}, 15}},
      {{10, 6}, {{22016, 404}, 11}},
      {{-12, 1}, {{0.000006139278412, 2.71875}, 15}},
  });
  testH2exp10Cases({
      {{-0.3, -0.6}, {{0.5, 0.25}, 16}},
      {{0.34, 0.7}, {{2.1875, 5}, 15}},
      {{0.8, 2}, {{6.3125, 100}, 14}},
      {{4, 3}, {{9984, 1000}, 12}},
      {{-12, 1}, {{0.000000000001002, 10}, 15}},
  });
  testH2exp2Cases({
      {{-0.3, -0.6}, {{0.8125, 0.66015625}, 16}},
      {{0.34, 0.7}, {{1.265625, 1.625}, 15}},
      {{0.8, 2}, {{1.7421875, 4}, 15}},
      {{12, 6}, {{4096, 64}, 12}},
      {{-12, 1}, {{0.000244140625, 2}, 15}},
  });
  testH2floorCases({
      {{-0.3, -0.6}, {{-1, -1}, 15}},
      {{0.34, 0.7}, {{0, 0}, 37}},
      {{0.8, 2}, {{0, 2}, 15}},
      {{23, 6}, {{23, 6}, 14}},
      {{-12, 1}, {{-12, 1}, 15}},
  });
  testH2logCases({
      {{0.3, 0.6}, {{-1.203125, -0.5078125}, 15}},
      {{0.34, 0.7}, {{-1.078125, -0.357421875}, 15}},
      {{0.8, 2}, {{-0.222, 0.691}, 3}},
      {{23, 6}, {{3.140625, 1.7890625}, 15}},
      {{12, 1}, {{2.484375, 0}, 15}},
  });
  testH2log10Cases({
      {{0.3, 0.6}, {{-0.5234375, -0.220703125}, 16}},
      {{0.34, 0.7}, {{-0.46875, -0.1552734375}, 16}},
      {{0.8, 2}, {{-0.0966796875, 0.30078125}, 16}},
      {{23, 6}, {{1.359375, 0.77734375}, 15}},
      {{12, 1}, {{1.078125, 0}, 15}},
  });
  testH2log2Cases({
      {{0.3, 0.6}, {{-1.734375, -0.734375}, 15}},
      {{0.34, 0.7}, {{-1.5546875, -0.515625}, 15}},
      {{0.8, 2}, {{-0.3203125, 1}, 15}},
      {{23, 6}, {{4.53125, 2.578125}, 15}},
      {{12, 1}, {{3.578125, 0}, 15}},
  });
  testH2rcpCases({
      {{-0.3, -0.6}, {{-3.328125, -1.6640625}, 15}},
      {{0.34, 0.7}, {{2.9375, 1.4296875}, 15}},
      {{0.8, 2}, {{1.25, 0.5}, 15}},
      {{23, 6}, {{0.04345703125, 0.1669921875}, 16}},
      {{-12, 1}, {{-0.08349609375, 1}, 15}},
  });
  testH2rintCases({
      {{-0.3, -0.6}, {{0, -1}, 15}},
      {{0.34, 0.7}, {{0, 1}, 15}},
      {{0.8, 2}, {{1, 2}, 15}},
      {{23, 6}, {{23, 6}, 14}},
      {{-12, 1}, {{-12, 1}, 14}},
  });
  testH2rsqrtCases({
      {{0.3, 0.6}, {{1.8203125, 1.2890625}, 15}},
      {{0.34, 0.7}, {{1.71875, 1.1953125}, 15}},
      {{0.8, 2}, {{1.1171875, 0.70703125}, 15}},
      {{23, 6}, {{0.209, 0.408}, 3}},
      {{12, 1}, {{0.2890625, 1}, 15}},
  });
  testH2sinCases({
      {{-0.3, -0.6}, {{-0.296875, -0.56640625}, 16}},
      {{0.34, 0.7}, {{0.333984375, 0.64453125}, 16}},
      {{0.8, 2}, {{0.71875, 0.91015625}, 16}},
      {{23, 6}, {{-0.84765625, -0.279296875}, 16}},
      {{-12, 1}, {{0.53515625, 0.83984375}, 16}},
  });
  testH2sqrtCases({
      {{0.3, 0.6}, {{0.546875, 0.77734375}, 16}},
      {{0.34, 0.7}, {{0.58203125, 0.8359375}, 16}},
      {{0.8, 2}, {{0.89453125, 1.4140625}, 15}},
      {{23, 6}, {{4.78125, 2.453125}, 15}},
      {{12, 1}, {{3.46875, 1}, 15}},
  });
  testH2truncCases({
      {{-0.3, -0.6}, {{0, 0}, 37}},
      {{0.34, 0.7}, {{0, 0}, 37}},
      {{0.8, 2}, {{0, 2}, 15}},
      {{23, 6}, {{23, 6}, 14}},
      {{-12, 1}, {{-12, 1}, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
