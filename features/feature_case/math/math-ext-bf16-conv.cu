// ====---------- math-ext-bf16-conv.cu---------- *- CUDA -* --------------===//
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

typedef pair<float2, int> f2i_pair;
typedef pair<float, int> fi_pair;
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

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat16 &Expect, const float &Result,
                 const int precision) {
  float FExpect = __bfloat162float(Expect);
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat16> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__bfloat162float(it));
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

__global__ void bfloat1622float2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __bfloat1622float2(Input1);
  Result[0] = ret.x;
  Result[1] = ret.y;
}

void testBfloat1622float2Cases(
    const vector<pair<__nv_bfloat162, f2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat1622float2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat1622float2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    auto ret = __bfloat1622float2(TestCase.first);
    Result[0] = ret.x;
    Result[1] = ret.y;
    checkResult("(host)__bfloat1622float2", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void bfloat162float(float *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162float(Input1);
}

void testBfloat162floatCases(
    const vector<pair<__nv_bfloat16, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162float<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162float", vector<__nv_bfloat16>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __bfloat162float(TestCase.first);
    checkResult("(host)__bfloat162float", vector<__nv_bfloat16>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void bfloat162int_rd(int *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162int_rd(Input1);
}

void testBfloat162int_rdCases(
    const vector<pair<__nv_bfloat16, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162int_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162int_rd", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162int_rn(int *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162int_rn(Input1);
}

void testBfloat162int_rnCases(
    const vector<pair<__nv_bfloat16, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162int_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162int_rn", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162int_ru(int *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162int_ru(Input1);
}

void testBfloat162int_ruCases(
    const vector<pair<__nv_bfloat16, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162int_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162int_ru", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162int_rz(int *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162int_rz(Input1);
}

void testBfloat162int_rzCases(
    const vector<pair<__nv_bfloat16, int>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162int_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162int_rz", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ll_rd(long long *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162ll_rd(Input1);
}

void testBfloat162ll_rdCases(
    const vector<pair<__nv_bfloat16, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ll_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ll_rd", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void bfloat162ll_rn(long long *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162ll_rn(Input1);
}

void testBfloat162ll_rnCases(
    const vector<pair<__nv_bfloat16, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ll_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ll_rn", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void bfloat162ll_ru(long long *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162ll_ru(Input1);
}

void testBfloat162ll_ruCases(
    const vector<pair<__nv_bfloat16, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ll_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ll_ru", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void bfloat162ll_rz(long long *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162ll_rz(Input1);
}

void testBfloat162ll_rzCases(
    const vector<pair<__nv_bfloat16, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ll_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ll_rz", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void bfloat162short_rd(short *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162short_rd(Input1);
}

void testBfloat162short_rdCases(
    const vector<pair<__nv_bfloat16, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162short_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162short_rd", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162short_rn(short *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162short_rn(Input1);
}

void testBfloat162short_rnCases(
    const vector<pair<__nv_bfloat16, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162short_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162short_rn", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162short_ru(short *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162short_ru(Input1);
}

void testBfloat162short_ruCases(
    const vector<pair<__nv_bfloat16, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162short_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162short_ru", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162short_rz(short *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162short_rz(Input1);
}

void testBfloat162short_rzCases(
    const vector<pair<__nv_bfloat16, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162short_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162short_rz", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162uint_rd(unsigned *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162uint_rd(Input1);
}

void testBfloat162uint_rdCases(
    const vector<pair<__nv_bfloat16, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162uint_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162uint_rd", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162uint_rn(unsigned *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162uint_rn(Input1);
}

void testBfloat162uint_rnCases(
    const vector<pair<__nv_bfloat16, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162uint_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162uint_rn", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162uint_ru(unsigned *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162uint_ru(Input1);
}

void testBfloat162uint_ruCases(
    const vector<pair<__nv_bfloat16, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162uint_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162uint_ru", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162uint_rz(unsigned *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162uint_rz(Input1);
}

void testBfloat162uint_rzCases(
    const vector<pair<__nv_bfloat16, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162uint_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162uint_rz", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ull_rd(unsigned long long *const Result,
                                __nv_bfloat16 Input1) {
  *Result = __bfloat162ull_rd(Input1);
}

void testBfloat162ull_rdCases(
    const vector<pair<__nv_bfloat16, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ull_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ull_rd", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ull_rn(unsigned long long *const Result,
                                __nv_bfloat16 Input1) {
  *Result = __bfloat162ull_rn(Input1);
}

void testBfloat162ull_rnCases(
    const vector<pair<__nv_bfloat16, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ull_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ull_rn", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ull_ru(unsigned long long *const Result,
                                __nv_bfloat16 Input1) {
  *Result = __bfloat162ull_ru(Input1);
}

void testBfloat162ull_ruCases(
    const vector<pair<__nv_bfloat16, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ull_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ull_ru", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ull_rz(unsigned long long *const Result,
                                __nv_bfloat16 Input1) {
  *Result = __bfloat162ull_rz(Input1);
}

void testBfloat162ull_rzCases(
    const vector<pair<__nv_bfloat16, unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ull_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ull_rz", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ushort_rd(unsigned short *const Result,
                                   __nv_bfloat16 Input1) {
  *Result = __bfloat162ushort_rd(Input1);
}

void testBfloat162ushort_rdCases(
    const vector<pair<__nv_bfloat16, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ushort_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ushort_rd", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ushort_rn(unsigned short *const Result,
                                   __nv_bfloat16 Input1) {
  *Result = __bfloat162ushort_rn(Input1);
}

void testBfloat162ushort_rnCases(
    const vector<pair<__nv_bfloat16, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ushort_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ushort_rn", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ushort_ru(unsigned short *const Result,
                                   __nv_bfloat16 Input1) {
  *Result = __bfloat162ushort_ru(Input1);
}

void testBfloat162ushort_ruCases(
    const vector<pair<__nv_bfloat16, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ushort_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ushort_ru", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat162ushort_rz(unsigned short *const Result,
                                   __nv_bfloat16 Input1) {
  *Result = __bfloat162ushort_rz(Input1);
}

void testBfloat162ushort_rzCases(
    const vector<pair<__nv_bfloat16, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162ushort_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162ushort_rz", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat16_as_short(short *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat16_as_short(Input1);
}

void testBfloat16_as_shortCases(
    const vector<pair<__nv_bfloat16, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat16_as_short<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat16_as_short", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void bfloat16_as_ushort(unsigned short *const Result,
                                   __nv_bfloat16 Input1) {
  *Result = __bfloat16_as_ushort(Input1);
}

void testBfloat16_as_ushortCases(
    const vector<pair<__nv_bfloat16, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat16_as_ushort<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat16_as_ushort", {TestCase.first}, TestCase.second,
                *Result);
  }
}

__global__ void float2bfloat16(float *const Result, float Input1) {
  *Result = __float2bfloat16(Input1);
}

void testFloat2bfloat16Cases(const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void float2bfloat16_rd(float *const Result, float Input1) {
  *Result = __float2bfloat16_rd(Input1);
}

void testFloat2bfloat16_rdCases(
    const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16_rd", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void float2bfloat16_rn(float *const Result, float Input1) {
  *Result = __float2bfloat16_rn(Input1);
}

void testFloat2bfloat16_rnCases(
    const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16_rn", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void float2bfloat16_ru(float *const Result, float Input1) {
  *Result = __float2bfloat16_ru(Input1);
}

void testFloat2bfloat16_ruCases(
    const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16_ru", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void float2bfloat16_rz(float *const Result, float Input1) {
  *Result = __float2bfloat16_rz(Input1);
}

void testFloat2bfloat16_rzCases(
    const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16_rz", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void int2bfloat16_rd(float *const Result, int Input1) {
  *Result = __int2bfloat16_rd(Input1);
}

void testInt2bfloat16_rdCases(const vector<pair<int, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2bfloat16_rn(float *const Result, int Input1) {
  *Result = __int2bfloat16_rn(Input1);
}

void testInt2bfloat16_rnCases(const vector<pair<int, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2bfloat16_ru(float *const Result, int Input1) {
  *Result = __int2bfloat16_ru(Input1);
}

void testInt2bfloat16_ruCases(const vector<pair<int, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void int2bfloat16_rz(float *const Result, int Input1) {
  *Result = __int2bfloat16_rz(Input1);
}

void testInt2bfloat16_rzCases(const vector<pair<int, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    int2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__int2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2bfloat16_rd(float *const Result, long long Input1) {
  *Result = __ll2bfloat16_rd(Input1);
}

void testLl2bfloat16_rdCases(
    const vector<pair<long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2bfloat16_rn(float *const Result, long long Input1) {
  *Result = __ll2bfloat16_rn(Input1);
}

void testLl2bfloat16_rnCases(
    const vector<pair<long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2bfloat16_ru(float *const Result, long long Input1) {
  *Result = __ll2bfloat16_ru(Input1);
}

void testLl2bfloat16_ruCases(
    const vector<pair<long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ll2bfloat16_rz(float *const Result, long long Input1) {
  *Result = __ll2bfloat16_rz(Input1);
}

void testLl2bfloat16_rzCases(
    const vector<pair<long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ll2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ll2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2bfloat16_rd(float *const Result, short Input1) {
  *Result = __short2bfloat16_rd(Input1);
}

void testShort2bfloat16_rdCases(
    const vector<pair<short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2bfloat16_rn(float *const Result, short Input1) {
  *Result = __short2bfloat16_rn(Input1);
}

void testShort2bfloat16_rnCases(
    const vector<pair<short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2bfloat16_ru(float *const Result, short Input1) {
  *Result = __short2bfloat16_ru(Input1);
}

void testShort2bfloat16_ruCases(
    const vector<pair<short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short2bfloat16_rz(float *const Result, short Input1) {
  *Result = __short2bfloat16_rz(Input1);
}

void testShort2bfloat16_rzCases(
    const vector<pair<short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void short_as_bfloat16(float *const Result, short Input1) {
  *Result = __short_as_bfloat16(Input1);
}

void testShort_as_bfloat16Cases(
    const vector<pair<short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short_as_bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short_as_bfloat16", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2bfloat16_rd(float *const Result, unsigned Input1) {
  *Result = __uint2bfloat16_rd(Input1);
}

void testUint2bfloat16_rdCases(
    const vector<pair<unsigned, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2bfloat16_rn(float *const Result, unsigned Input1) {
  *Result = __uint2bfloat16_rn(Input1);
}

void testUint2bfloat16_rnCases(
    const vector<pair<unsigned, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2bfloat16_ru(float *const Result, unsigned Input1) {
  *Result = __uint2bfloat16_ru(Input1);
}

void testUint2bfloat16_ruCases(
    const vector<pair<unsigned, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void uint2bfloat16_rz(float *const Result, unsigned Input1) {
  *Result = __uint2bfloat16_rz(Input1);
}

void testUint2bfloat16_rzCases(
    const vector<pair<unsigned, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    uint2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__uint2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2bfloat16_rd(float *const Result,
                                unsigned long long Input1) {
  *Result = __ull2bfloat16_rd(Input1);
}

void testUll2bfloat16_rdCases(
    const vector<pair<unsigned long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2bfloat16_rn(float *const Result,
                                unsigned long long Input1) {
  *Result = __ull2bfloat16_rn(Input1);
}

void testUll2bfloat16_rnCases(
    const vector<pair<unsigned long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2bfloat16_ru(float *const Result,
                                unsigned long long Input1) {
  *Result = __ull2bfloat16_ru(Input1);
}

void testUll2bfloat16_ruCases(
    const vector<pair<unsigned long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ull2bfloat16_rz(float *const Result,
                                unsigned long long Input1) {
  *Result = __ull2bfloat16_rz(Input1);
}

void testUll2bfloat16_rzCases(
    const vector<pair<unsigned long long, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ull2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ull2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2bfloat16_rd(float *const Result, unsigned short Input1) {
  *Result = __ushort2bfloat16_rd(Input1);
}

void testUshort2bfloat16_rdCases(
    const vector<pair<unsigned short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2bfloat16_rd<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2bfloat16_rd", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2bfloat16_rn(float *const Result, unsigned short Input1) {
  *Result = __ushort2bfloat16_rn(Input1);
}

void testUshort2bfloat16_rnCases(
    const vector<pair<unsigned short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2bfloat16_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2bfloat16_rn", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2bfloat16_ru(float *const Result, unsigned short Input1) {
  *Result = __ushort2bfloat16_ru(Input1);
}

void testUshort2bfloat16_ruCases(
    const vector<pair<unsigned short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2bfloat16_ru<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2bfloat16_ru", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort2bfloat16_rz(float *const Result, unsigned short Input1) {
  *Result = __ushort2bfloat16_rz(Input1);
}

void testUshort2bfloat16_rzCases(
    const vector<pair<unsigned short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort2bfloat16_rz<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort2bfloat16_rz", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void ushort_as_bfloat16(float *const Result, unsigned short Input1) {
  *Result = __ushort_as_bfloat16(Input1);
}

void testUshort_as_bfloat16Cases(
    const vector<pair<unsigned short, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort_as_bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort_as_bfloat16", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testBfloat1622float2Cases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testBfloat162floatCases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
  });
  testBfloat162int_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162int_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162int_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162int_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ll_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ll_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ll_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ll_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162short_rdCases({
      {-0.3, -1},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162short_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162short_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162short_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162uint_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162uint_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162uint_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162uint_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ull_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ull_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ull_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ull_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ushort_rdCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat162ushort_rnCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ushort_ruCases({
      {-0.3, 0},
      {0.3, 1},
      {30, 30},
      {0.432643, 1},
      {1, 1},
      {10.7, 11},
  });
  testBfloat162ushort_rzCases({
      {-0.3, 0},
      {0.3, 0},
      {30, 30},
      {0.432643, 0},
      {1, 1},
      {10.7, 10},
  });
  testBfloat16_as_shortCases({
      {-0.3, -16742},
      {0.3, 16026},
      {30, 16880},
      {0.432643, 16094},
      {1, 16256},
      {10.7, 16683},
  });
  testBfloat16_as_ushortCases({
      {-0.3, 48794},
      {0.3, 16026},
      {30, 16880},
      {0.432643, 16094},
      {1, 16256},
      {10.7, 16683},
  });
  testFloat2bfloat16Cases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  testFloat2bfloat16_rdCases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.298828125, 16}},
      {30, {30, 14}},
      {0.432643, {0.431640625, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  testFloat2bfloat16_rnCases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  testFloat2bfloat16_ruCases({
      {-0.3, {-0.298828125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
      {1, {1, 15}},
      {10.7, {10.75, 15}},
  });
  testFloat2bfloat16_rzCases({
      {-0.3, {-0.298828125, 16}},
      {0.3, {0.298828125, 16}},
      {30, {30, 14}},
      {0.432643, {0.431640625, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  testInt2bfloat16_rdCases({
      {-10000, {-10048, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testInt2bfloat16_rnCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testInt2bfloat16_ruCases({
      {-10000, {-9984, 12}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testInt2bfloat16_rzCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testLl2bfloat16_rdCases({
      {-10000, {-10048, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testLl2bfloat16_rnCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testLl2bfloat16_ruCases({
      {-10000, {-9984, 12}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testLl2bfloat16_rzCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testShort2bfloat16_rdCases({
      {-10000, {-10048, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testShort2bfloat16_rnCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-3008, 12}},
  });
  testShort2bfloat16_ruCases({
      {-10000, {-9984, 12}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testShort2bfloat16_rzCases({
      {-10000, {-9984, 12}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {-3000, {-2992, 12}},
  });
  testShort_as_bfloat16Cases({
      {-10000, {-2111062325329920.0, 0}},
      {10000, {0.000000000000001998401444325282, 30}},
      {30000, {223106505640168374663419764146176.0, -17}},
      {3000, {0.00000000000000000000000000000007087422195345028, 47}},
      {1000, {0.0000000000000000000000000000000000013635734469538535, 52}},
      {-3000, {-63382530011411470074835160268800.0, -16}},
  });
  testUint2bfloat16_rdCases({
      {4294957296, {4278190080.0, 6}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {4294964296, {4278190080.0, 6}},
  });
  testUint2bfloat16_rnCases({
      {4294957296, {4294967296.0, 6}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {4294964296, {4294967296.0, 6}},
  });
  testUint2bfloat16_ruCases({
      {4294957296, {4294967296.0, 6}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {4294964296, {4294967296.0, 6}},
  });
  testUint2bfloat16_rzCases({
      {4294957296, {4278190080.0, 6}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {4294964296, {4278190080.0, 6}},
  });
  testUll2bfloat16_rdCases({
      {18446744073709541616, {18374686479671623680.0, -4}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {18446744073709548616, {18374686479671623680.0, -4}},
  });
  testUll2bfloat16_rnCases({
      {18446744073709541616, {18446744073709551616.0, -4}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {18446744073709548616, {18446744073709551616.0, -4}},
  });
  testUll2bfloat16_ruCases({
      {18446744073709541616, {18446744073709551616.0, -4}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {18446744073709548616, {18446744073709551616.0, -4}},
  });
  testUll2bfloat16_rzCases({
      {18446744073709541616, {18374686479671623680.0, -4}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {18446744073709548616, {18374686479671623680.0, -4}},
  });
  testUshort2bfloat16_rdCases({
      {55536, {55296.0, 11}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {62536, {62464.0, 11}},
  });
  testUshort2bfloat16_rnCases({
      {55536, {55552.0, 11}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {62536, {62464.0, 11}},
  });
  testUshort2bfloat16_ruCases({
      {55536, {55552.0, 11}},
      {10000, {10048, 12}},
      {30000, {30080, 11}},
      {3000, {3008, 12}},
      {1000, {1000, 13}},
      {62536, {62720.0, 11}},
  });
  testUshort2bfloat16_rzCases({
      {55536, {55296.0, 11}},
      {10000, {9984, 12}},
      {30000, {29952, 11}},
      {3000, {2992, 12}},
      {1000, {1000, 13}},
      {62536, {62464.0, 11}},
  });
  testUshort_as_bfloat16Cases({
      {55536, {-2111062325329920.0, 0}},
      {10000, {0.0000000000000019984014443252817727625, 30}},
      {30000, {223106505640168374663419764146176.0, -17}},
      {3000, {0.00000000000000000000000000000007087422195345028, 47}},
      {1000, {0.0000000000000000000000000000000000013635734469538535, 52}},
      {62536, {-63382530011411470074835160268800.0, -16}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
