// ===------------- math-ext-half-conv.cu--------------- *- CUDA -* -------===//
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
typedef pair<__half2, int> h2i_pair;
typedef pair<float, int> fi_pair;
typedef pair<float2, int> f2i_pair;

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
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
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

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
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

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const int &Expect, const int &Result) {
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

void checkResult(const string &FuncName, const vector<int> &Inputs,
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

void checkResult(const string &FuncName, const vector<__half> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__half2float(it));
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  float2 FExpect{__half2float(Expect.x), __half2float(Expect.y)};
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__half2> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__half2float(it.x), __half2float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  float2 FExpect{__half2float(Expect.x), __half2float(Expect.y)};
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

// Half Precision Conversion and Data Movement

__global__ void float22half2_rn(float *const Result, float2 Input1) {
  auto ret = __float22half2_rn(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testFloat22half2_rnCases(const vector<pair<float2, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    float22half2_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float22half2_rn", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    auto ret = __float22half2_rn(TestCase.first);
    Result[0] = __half2float(ret.x);
    Result[1] = __half2float(ret.y);
    checkResult("(host)__float22half2_rn", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void float2half(float *const Result, float Input1) {
  *Result = __float2half(Input1);
}

void testFloat2halfCases(const vector<pair<float, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __float2half(TestCase.first);
    checkResult("(host)__float2half", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float2half2_rn(float *const Result, float Input1) {
  auto ret = __float2half2_rn(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testFloat2half2_rnCases(const vector<pair<float, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    float2half2_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half2_rn", vector<float>{TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret = __float2half2_rn(TestCase.first);
    Result[0] = __half2float(ret.x);
    Result[1] = __half2float(ret.y);
    checkResult("(host)__float2half2_rn", vector<float>{TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void float2half_rd(float *const Result, float Input1) {
  *Result = __float2half_rd(Input1);
}

void testFloat2half_rdCases(const vector<pair<float, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half_rd", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __float2half_rd(TestCase.first);
    checkResult("(host)__float2half_rd", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float2half_rn(float *const Result, float Input1) {
  *Result = __float2half_rn(Input1);
}

void testFloat2half_rnCases(const vector<pair<float, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half_rn", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __float2half_rn(TestCase.first);
    checkResult("(host)__float2half_rn", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float2half_ru(float *const Result, float Input1) {
  *Result = __float2half_ru(Input1);
}

void testFloat2half_ruCases(const vector<pair<float, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half_ru", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __float2half_ru(TestCase.first);
    checkResult("(host)__float2half_ru", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float2half_rz(float *const Result, float Input1) {
  *Result = __float2half_rz(Input1);
}

void testFloat2half_rzCases(const vector<pair<float, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2half_rz", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __float2half_rz(TestCase.first);
    checkResult("(host)__float2half_rz", vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void floats2half2_rn(float *const Result, float Input1,
                                float Input2) {
  auto ret = __floats2half2_rn(Input1, Input2);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testFloats2half2_rnCases(
    const vector<pair<pair<float, float>, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    floats2half2_rn<<<1, 1>>>(Result, TestCase.first.first,
                              TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__floats2half2_rn",
                vector<float>{TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret = __floats2half2_rn(TestCase.first.first, TestCase.first.second);
    Result[0] = __half2float(ret.x);
    Result[1] = __half2float(ret.y);
    checkResult("(host)__floats2half2_rn",
                vector<float>{TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void half22float2(float *const Result, __half2 Input1) {
  auto ret = __half22float2(Input1);
  Result[0] = ret.x;
  Result[1] = ret.y;
}

void testHalf22float2Cases(const vector<pair<__half2, f2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    half22float2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half22float2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    auto ret = __half22float2(TestCase.first);
    Result[0] = ret.x;
    Result[1] = ret.y;
    checkResult("(host)__half22float2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void half2float(float *const Result, __half Input1) {
  *Result = __half2float(Input1);
}

void testHalf2floatCases(const vector<pair<__half, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2float<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2float", vector<__half>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __half2float(TestCase.first);
    checkResult("(host)__half2float", vector<__half>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void half2int_rd(int *const Result, __half Input1) {
  *Result = __half2int_rd(Input1);
}

void testHalf2int_rdCases(const vector<pair<__half, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2int_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2int_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2int_rn(int *const Result, __half Input1) {
  *Result = __half2int_rn(Input1);
}

void testHalf2int_rnCases(const vector<pair<__half, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2int_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2int_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2int_ru(int *const Result, __half Input1) {
  *Result = __half2int_ru(Input1);
}

void testHalf2int_ruCases(const vector<pair<__half, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2int_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2int_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2int_rz(int *const Result, __half Input1) {
  *Result = __half2int_rz(Input1);
}

void testHalf2int_rzCases(const vector<pair<__half, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2int_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2int_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ll_rd(long long *const Result, __half Input1) {
  *Result = __half2ll_rd(Input1);
}

void testHalf2ll_rdCases(const vector<pair<__half, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ll_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ll_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ll_rn(long long *const Result, __half Input1) {
  *Result = __half2ll_rn(Input1);
}

void testHalf2ll_rnCases(const vector<pair<__half, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ll_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ll_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ll_ru(long long *const Result, __half Input1) {
  *Result = __half2ll_ru(Input1);
}

void testHalf2ll_ruCases(const vector<pair<__half, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ll_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ll_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ll_rz(long long *const Result, __half Input1) {
  *Result = __half2ll_rz(Input1);
}

void testHalf2ll_rzCases(const vector<pair<__half, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ll_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ll_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2short_rd(short *const Result, __half Input1) {
  *Result = __half2short_rd(Input1);
}

void testHalf2short_rdCases(const vector<pair<__half, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2short_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2short_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2short_rn(short *const Result, __half Input1) {
  *Result = __half2short_rn(Input1);
}

void testHalf2short_rnCases(const vector<pair<__half, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2short_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2short_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2short_ru(short *const Result, __half Input1) {
  *Result = __half2short_ru(Input1);
}

void testHalf2short_ruCases(const vector<pair<__half, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2short_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2short_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2short_rz(short *const Result, __half Input1) {
  *Result = __half2short_rz(Input1);
}

void testHalf2short_rzCases(const vector<pair<__half, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2short_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2short_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2uint_rd(unsigned *const Result, __half Input1) {
  *Result = __half2uint_rd(Input1);
}

void testHalf2uint_rdCases(const vector<pair<__half, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2uint_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2uint_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2uint_rn(unsigned *const Result, __half Input1) {
  *Result = __half2uint_rn(Input1);
}

void testHalf2uint_rnCases(const vector<pair<__half, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2uint_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2uint_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2uint_ru(unsigned *const Result, __half Input1) {
  *Result = __half2uint_ru(Input1);
}

void testHalf2uint_ruCases(const vector<pair<__half, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2uint_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2uint_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2uint_rz(unsigned *const Result, __half Input1) {
  *Result = __half2uint_rz(Input1);
}

void testHalf2uint_rzCases(const vector<pair<__half, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2uint_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2uint_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ull_rd(unsigned long long *const Result, __half Input1) {
  *Result = __half2ull_rd(Input1);
}

void testHalf2ull_rdCases(
    const vector<pair<__half, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ull_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ull_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ull_rn(unsigned long long *const Result, __half Input1) {
  *Result = __half2ull_rn(Input1);
}

void testHalf2ull_rnCases(
    const vector<pair<__half, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ull_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ull_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ull_ru(unsigned long long *const Result, __half Input1) {
  *Result = __half2ull_ru(Input1);
}

void testHalf2ull_ruCases(
    const vector<pair<__half, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ull_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ull_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ull_rz(unsigned long long *const Result, __half Input1) {
  *Result = __half2ull_rz(Input1);
}

void testHalf2ull_rzCases(
    const vector<pair<__half, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ull_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ull_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ushort_rd(unsigned short *const Result, __half Input1) {
  *Result = __half2ushort_rd(Input1);
}

void testHalf2ushort_rdCases(
    const vector<pair<__half, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ushort_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ushort_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ushort_rn(unsigned short *const Result, __half Input1) {
  *Result = __half2ushort_rn(Input1);
}

void testHalf2ushort_rnCases(
    const vector<pair<__half, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ushort_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ushort_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ushort_ru(unsigned short *const Result, __half Input1) {
  *Result = __half2ushort_ru(Input1);
}

void testHalf2ushort_ruCases(
    const vector<pair<__half, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ushort_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ushort_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half2ushort_rz(unsigned short *const Result, __half Input1) {
  *Result = __half2ushort_rz(Input1);
}

void testHalf2ushort_rzCases(
    const vector<pair<__half, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2ushort_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2ushort_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void int2half_rd(float *const Result, int Input1) {
  *Result = __int2half_rd(Input1);
}

void testInt2half_rdCases(const vector<pair<int, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2half_rd", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2half_rn(float *const Result, int Input1) {
  *Result = __int2half_rn(Input1);
}

void testInt2half_rnCases(const vector<pair<int, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2half_rn", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2half_ru(float *const Result, int Input1) {
  *Result = __int2half_ru(Input1);
}

void testInt2half_ruCases(const vector<pair<int, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2half_ru", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2half_rz(float *const Result, int Input1) {
  *Result = __int2half_rz(Input1);
}

void testInt2half_rzCases(const vector<pair<int, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2half_rz", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2half_rd(float *const Result, long long Input1) {
  *Result = __ll2half_rd(Input1);
}

void testLl2half_rdCases(const vector<pair<long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2half_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2half_rn(float *const Result, long long Input1) {
  *Result = __ll2half_rn(Input1);
}

void testLl2half_rnCases(const vector<pair<long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2half_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2half_ru(float *const Result, long long Input1) {
  *Result = __ll2half_ru(Input1);
}

void testLl2half_ruCases(const vector<pair<long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2half_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2half_rz(float *const Result, long long Input1) {
  *Result = __ll2half_rz(Input1);
}

void testLl2half_rzCases(const vector<pair<long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2half_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2half_rd(float *const Result, short Input1) {
  *Result = __short2half_rd(Input1);
}

void testShort2half_rdCases(const vector<pair<short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2half_rd", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2half_rn(float *const Result, short Input1) {
  *Result = __short2half_rn(Input1);
}

void testShort2half_rnCases(const vector<pair<short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2half_rn", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2half_ru(float *const Result, short Input1) {
  *Result = __short2half_ru(Input1);
}

void testShort2half_ruCases(const vector<pair<short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2half_ru", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2half_rz(float *const Result, short Input1) {
  *Result = __short2half_rz(Input1);
}

void testShort2half_rzCases(const vector<pair<short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2half_rz", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2half_rd(float *const Result, unsigned Input1) {
  *Result = __uint2half_rd(Input1);
}

void testUint2half_rdCases(const vector<pair<unsigned, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2half_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2half_rn(float *const Result, unsigned Input1) {
  *Result = __uint2half_rn(Input1);
}

void testUint2half_rnCases(const vector<pair<unsigned, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2half_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2half_ru(float *const Result, unsigned Input1) {
  *Result = __uint2half_ru(Input1);
}

void testUint2half_ruCases(const vector<pair<unsigned, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2half_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2half_rz(float *const Result, unsigned Input1) {
  *Result = __uint2half_rz(Input1);
}

void testUint2half_rzCases(const vector<pair<unsigned, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2half_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2half_rd(float *const Result, unsigned long long Input1) {
  *Result = __ull2half_rd(Input1);
}

void testUll2half_rdCases(
    const vector<pair<unsigned long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2half_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2half_rn(float *const Result, unsigned long long Input1) {
  *Result = __ull2half_rn(Input1);
}

void testUll2half_rnCases(
    const vector<pair<unsigned long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2half_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2half_ru(float *const Result, unsigned long long Input1) {
  *Result = __ull2half_ru(Input1);
}

void testUll2half_ruCases(
    const vector<pair<unsigned long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2half_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2half_rz(float *const Result, unsigned long long Input1) {
  *Result = __ull2half_rz(Input1);
}

void testUll2half_rzCases(
    const vector<pair<unsigned long long, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2half_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2half_rd(float *const Result, unsigned short Input1) {
  *Result = __ushort2half_rd(Input1);
}

void testUshort2half_rdCases(
    const vector<pair<unsigned short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2half_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2half_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2half_rn(float *const Result, unsigned short Input1) {
  *Result = __ushort2half_rn(Input1);
}

void testUshort2half_rnCases(
    const vector<pair<unsigned short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2half_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2half_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2half_ru(float *const Result, unsigned short Input1) {
  *Result = __ushort2half_ru(Input1);
}

void testUshort2half_ruCases(
    const vector<pair<unsigned short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2half_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2half_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2half_rz(float *const Result, unsigned short Input1) {
  *Result = __ushort2half_rz(Input1);
}

void testUshort2half_rzCases(
    const vector<pair<unsigned short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2half_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2half_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testFloat22half2_rnCases({
      {{-0.3, -0.4}, {{-0.3, -0.3999}, 4}},
      {{0, 0.7}, {{0, 0.7}, 3}},
      {{1, 100.6}, {{1, 100.6}, 1}},
      {{100.6, 1}, {{100.6, 1}, 1}},
  });
  testFloat2halfCases({
      {-0.3, {-0.3, 4}},
      {0.3, {0.3, 4}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testFloat2half2_rnCases({
      {-0.3, {{-0.3, -0.3}, 4}},
      {0.3, {{0.3, 0.3}, 4}},
      {0.5, {{0.5, 0.5}, 16}},
      {23, {{23, 23}, 14}},
  });
  testFloat2half_rdCases({
      {-0.3, {-0.300048828125, 16}},
      {0.3, {0.3, 3}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testFloat2half_rnCases({
      {-0.3, {-0.3, 4}},
      {0.3, {0.3, 4}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testFloat2half_ruCases({
      {-0.3, {-0.3, 3}},
      {0.3, {0.300048828125, 16}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testFloat2half_rzCases({
      {-0.3, {-0.3, 3}},
      {0.3, {0.3, 3}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testFloats2half2_rnCases({
      {{-0.3, -0.4}, {{-0.3, -0.3999}, 4}},
      {{0, 0.7}, {{0, 0.7}, 3}},
      {{1, 100.6}, {{1, 100.6}, 1}},
      {{100.6, 1}, {{100.6, 1}, 1}},
  });
  testHalf22float2Cases({
      {{-0.3, -0.4}, {{-0.300048828125, -0.39990234375}, 16}},
      {{0, 0.7}, {{0, 0.7001953125}, 16}},
      {{1, 100.6}, {{1, 100.625}, 14}},
      {{100.6, 1}, {{100.625, 1}, 14}},
  });
  testHalf2floatCases({
      {-0.3, {-0.3000488281250000, 16}},
      {0.3, {0.3000488281250000, 16}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testHalf2int_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2int_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2int_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2int_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ll_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ll_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ll_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ll_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2short_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2short_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2short_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2short_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2uint_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2uint_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2uint_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2uint_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ull_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ull_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ull_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ull_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ushort_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testHalf2ushort_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ushort_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testHalf2ushort_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testInt2half_rdCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testInt2half_rnCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testInt2half_ruCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testInt2half_rzCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testLl2half_rdCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testLl2half_rnCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testLl2half_ruCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testLl2half_rzCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testShort2half_rdCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testShort2half_rnCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testShort2half_ruCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testShort2half_rzCases({
      {-10000, {-10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {-3000, {-3000, 12}},
  });
  testUint2half_rdCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUint2half_rnCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUint2half_ruCases({
      {10001, {10008, -1}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUint2half_rzCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUll2half_rdCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUll2half_rnCases({
      {10001, {10000, -1}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUll2half_ruCases({
      {10001, {10008, -1}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUll2half_rzCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUshort2half_rdCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUshort2half_rnCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUshort2half_ruCases({
      {10001, {10008, -1}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  testUshort2half_rzCases({
      {10001, {10000, 12}},
      {10000, {10000, 12}},
      {30000, {30000, 11}},
      {3000, {3000, 12}},
      {1000, {1000, 13}},
      {300, {300, 12}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
