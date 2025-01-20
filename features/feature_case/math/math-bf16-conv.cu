// ====---------- math-bf16-conv.cu---------- *- CUDA -* ------------------===//
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
typedef pair<__nv_bfloat162, int> bf162i_pair;
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
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
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
                 const float &Expect, const float &Result,
                 const int precision) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
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

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat16 &Expect, const float &Result,
                 const int precision) {
  float FExpect = __bfloat162float(Expect);
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat162 &Expect, const float2 &Result,
                 const int precision) {
  float2 FExpect{__bfloat162float(Expect.x), __bfloat162float(Expect.y)};
  checkResult(FuncName, Inputs, FExpect, Result, precision);
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

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const __nv_bfloat162 &Expect, const float2 &Result,
                 const int precision) {
  float2 FExpect{__bfloat162float(Expect.x), __bfloat162float(Expect.y)};
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const __nv_bfloat162 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

__global__ void setValue(__nv_bfloat16 *Input1, const __nv_bfloat16 Input2) {
  *Input1 = Input2;
}

__global__ void setValue(__nv_bfloat162 *Input1, const __nv_bfloat162 Input2) {
  *Input1 = Input2;
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

__global__ void bfloat162bfloat162(float *const Result, __nv_bfloat16 Input1) {
  auto ret = __bfloat162bfloat162(Input1);
  Result[0] = ret.x;
  Result[1] = ret.y;
}

void testBfloat162bfloat162Cases(
    const vector<pair<__nv_bfloat16, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bfloat162bfloat162<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162bfloat162", vector<float>{TestCase.first},
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
    checkResult("__bfloat162float", std::vector<float>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
    *Result = __bfloat162float(TestCase.first);
    checkResult("(host)__bfloat162float", std::vector<float>{TestCase.first},
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

__global__ void float22bfloat162_rn(float *const Result, float2 Input1) {
  auto ret = __float22bfloat162_rn(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testFloat22bfloat162_rnCases(
    const vector<pair<float2, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float22bfloat162_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float22bfloat162_rn", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret = __float22bfloat162_rn(TestCase.first);
    Result[0] = __bfloat162float(ret.x);
    Result[1] = __bfloat162float(ret.y);
    checkResult("(host)__float22bfloat162_rn", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void float2bfloat16(float *const Result, float Input1) {
  *Result = __bfloat162float(__float2bfloat16(Input1));
}

void testFloat2bfloat16Cases(const vector<pair<float, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2bfloat16(TestCase.first);
    checkResult("(host)__float2bfloat16", {TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float2bfloat162_rn(float *const Result, float Input1) {
  auto ret = __float2bfloat162_rn(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testFloat2bfloat162_rnCases(
    const vector<pair<float, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bfloat162_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat162_rn", vector<float>{TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret = __float2bfloat162_rn(TestCase.first);
    Result[0] = __bfloat162float(ret.x);
    Result[1] = __bfloat162float(ret.y);
    checkResult("(host)__float2bfloat162_rn", vector<float>{TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void floats2bfloat162_rn(float *const Result, float Input1,
                                    float Input2) {
  auto ret = __floats2bfloat162_rn(Input1, Input2);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testFloats2bfloat162_rnCases(
    const vector<pair<pair<float, float>, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    floats2bfloat162_rn<<<1, 1>>>(Result, TestCase.first.first,
                                  TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__floats2bfloat162_rn",
                vector<float>{TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret =
        __floats2bfloat162_rn(TestCase.first.first, TestCase.first.second);
    Result[0] = __bfloat162float(ret.x);
    Result[1] = __bfloat162float(ret.y);
    checkResult("(host)__floats2bfloat162_rn",
                vector<float>{TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void halves2bfloat162(float *const Result, __nv_bfloat16 Input1,
                                 __nv_bfloat16 Input2) {
  auto ret = __halves2bfloat162(Input1, Input2);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testHalves2bfloat162Cases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf162i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    halves2bfloat162<<<1, 1>>>(Result, TestCase.first.first,
                               TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult(
        "__halves2bfloat162", {TestCase.first.first, TestCase.first.second},
        TestCase.second.first, {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void high2bfloat16(float *const Result, __nv_bfloat162 Input1) {
  *Result = __bfloat162float(__high2bfloat16(Input1));
}

void testHigh2bfloat16Cases(
    const vector<pair<__nv_bfloat162, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    high2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__high2bfloat16", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void high2bfloat162(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __high2bfloat162(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testHigh2bfloat162Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    high2bfloat162<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__high2bfloat162", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void highs2bfloat162(float *const Result, __nv_bfloat162 Input1,
                                __nv_bfloat162 Input2) {
  auto ret = __highs2bfloat162(Input1, Input2);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testHighs2bfloat162Cases(
    const vector<pair<pair<__nv_bfloat162, __nv_bfloat162>, bf162i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    highs2bfloat162<<<1, 1>>>(Result, TestCase.first.first,
                              TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult(
        "__highs2bfloat162", {TestCase.first.first, TestCase.first.second},
        TestCase.second.first, {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void ldca(float *const Result, __nv_bfloat16 *Input1) {
  *Result = __ldca(Input1);
}

void testLdcaCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat16 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldca(float *const Result, __nv_bfloat162 *Input1) {
  auto ret = __ldca(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLdcaCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat162 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __nv_bfloat16 *Input1) {
  *Result = __ldcg(Input1);
}

void testLdcgCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat16 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __nv_bfloat162 *Input1) {
  auto ret = __ldcg(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLdcgCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat162 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __nv_bfloat16 *Input1) {
  *Result = __ldcs(Input1);
}

void testLdcsCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat16 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __nv_bfloat162 *Input1) {
  auto ret = __ldcs(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLdcsCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat162 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __nv_bfloat16 *Input1) {
  *Result = __ldcv(Input1);
}

void testLdcvCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat16 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __nv_bfloat162 *Input1) {
  auto ret = __ldcv(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLdcvCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat162 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __nv_bfloat16 *Input1) {
  *Result = __ldlu(Input1);
}

void testLdluCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat16 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __nv_bfloat162 *Input1) {
  auto ret = __ldlu(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLdluCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    __nv_bfloat162 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void low2bfloat16(float *const Result, __nv_bfloat162 Input1) {
  *Result = __bfloat162float(__low2bfloat16(Input1));
}

void testLow2bfloat16Cases(
    const vector<pair<__nv_bfloat162, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    low2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__low2bfloat16", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void low2bfloat162(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __low2bfloat162(Input1);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLow2bfloat162Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    low2bfloat162<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__low2bfloat162", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void lows2bfloat162(float *const Result, __nv_bfloat162 Input1,
                               __nv_bfloat162 Input2) {
  auto ret = __lows2bfloat162(Input1, Input2);
  Result[0] = __bfloat162float(ret.x);
  Result[1] = __bfloat162float(ret.y);
}

void testLows2bfloat162Cases(
    const vector<pair<pair<__nv_bfloat162, __nv_bfloat162>, bf162i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    lows2bfloat162<<<1, 1>>>(Result, TestCase.first.first,
                             TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult(
        "__lows2bfloat162", {TestCase.first.first, TestCase.first.second},
        TestCase.second.first, {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void stcg(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 *const Temp) {
  __stcg(Temp, Input1);
  *Result = __bfloat162float(*Temp);
}

void testStcgCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __nv_bfloat16 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcg(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 *const Temp) {
  __stcg(Temp, Input1);
  Result[0] = __bfloat162float(Temp->x);
  Result[1] = __bfloat162float(Temp->y);
}

void testStcgCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __nv_bfloat162 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stcs(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 *const Temp) {
  __stcs(Temp, Input1);
  *Result = __bfloat162float(*Temp);
}

void testStcsCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __nv_bfloat16 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcs(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 *const Temp) {
  __stcs(Temp, Input1);
  Result[0] = __bfloat162float(Temp->x);
  Result[1] = __bfloat162float(Temp->y);
}

void testStcsCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __nv_bfloat162 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwb(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 *const Temp) {
  __stwb(Temp, Input1);
  *Result = __bfloat162float(*Temp);
}

void testStwbCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __nv_bfloat16 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwb(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 *const Temp) {
  __stwb(Temp, Input1);
  Result[0] = __bfloat162float(Temp->x);
  Result[1] = __bfloat162float(Temp->y);
}

void testStwbCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __nv_bfloat162 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwt(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 *const Temp) {
  __stwt(Temp, Input1);
  *Result = __bfloat162float(*Temp);
}

void testStwtCases(const vector<pair<__nv_bfloat16, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __nv_bfloat16 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwt(float *const Result, __nv_bfloat162 Input1,
                     __nv_bfloat162 *const Temp) {
  __stwt(Temp, Input1);
  Result[0] = __bfloat162float(Temp->x);
  Result[1] = __bfloat162float(Temp->y);
}

void testStwtCases(const vector<pair<__nv_bfloat162, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __nv_bfloat162 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

int main() {
  testBfloat1622float2Cases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testBfloat162bfloat162Cases({
      {-0.3, {{-0.30078125, -0.30078125}, 16}},
      {0.5, {{0.5, 0.5}, 16}},
      {30, {{30, 30}, 14}},
      {0.432643, {{0.43359375, 0.43359375}, 16}},
      {1, {{1, 1}, 15}},
      {100.6, {{100.5, 100.5}, 14}},
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
      // {-0.3, 0},
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
      // {-0.3, 0},
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
      // {-0.3, 0},
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
  testFloat22bfloat162_rnCases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testFloat2bfloat16Cases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
  });
  testFloat2bfloat162_rnCases({
      {-0.3, {{-0.30078125, -0.30078125}, 16}},
      {0.5, {{0.5, 0.5}, 16}},
      {30, {{30, 30}, 14}},
      {0.432643, {{0.43359375, 0.43359375}, 16}},
  });
  testFloats2bfloat162_rnCases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testHalves2bfloat162Cases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
      {{1, 5000}, {{1, 4992}, 12}},
      {{10.7, 3000000}, {{10.6875, 2998272}, 9}},
  });
  testHigh2bfloat16Cases({
      {{-0.3, -0.5}, {-0.5, 16}},
      {{0.3, 0.5}, {0.5, 16}},
      {{30, 50}, {50, 14}},
      {{0.432643, 0.23654}, {0.236328125, 16}},
      {{1, 5000}, {4992, 12}},
      {{10.7, 3000000}, {2998272, 9}},
  });
  testHigh2bfloat162Cases({
      {{-0.3, -0.5}, {{-0.5, -0.5}, 16}},
      {{0.3, 0.5}, {{0.5, 0.5}, 16}},
      {{30, 50}, {{50, 50}, 14}},
      {{0.432643, 0.23654}, {{0.236328125, 0.236328125}, 16}},
      {{1, 5000}, {{4992, 4992}, 12}},
      {{10.7, 3000000}, {{2998272, 2998272}, 9}},
  });
  testHighs2bfloat162Cases({
      {{{-0.3, -0.5}, {10.7, 3000000}}, {{-0.5, 2998272}, 9}},
      {{{0.3, 0.5}, {-0.3, -0.5}}, {{0.5, -0.5}, 16}},
      {{{30, 50}, {0.3, 0.5}}, {{50, 0.5}, 14}},
      {{{0.432643, 0.23654}, {30, 50}}, {{0.236328125, 50}, 14}},
      {{{1, 5000}, {0.432643, 0.23654}}, {{4992, 0.236328125000}, 12}},
      {{{10.7, 3000000}, {1, 5000}}, {{2998272, 4992}, 9}},
  });
  testLdcaCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcaCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcgCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcsCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcsCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcvCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcvCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdluCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdluCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLow2bfloat16Cases({
      {{-0.3, -0.5}, {-0.30078125, 16}},
      {{0.3, 0.5}, {0.30078125, 16}},
      {{30, 50}, {30, 14}},
      {{0.432643, 0.23654}, {0.43359375, 16}},
      {{1, 5000}, {1, 15}},
      {{10.7, 3000000}, {10.6875, 15}},
  });
  testLow2bfloat162Cases({
      {{-0.3, -0.5}, {{-0.30078125, -0.30078125}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.30078125}, 16}},
      {{30, 50}, {{30, 30}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.43359375}, 16}},
      {{1, 5000}, {{1, 1}, 15}},
      {{10.7, 3000000}, {{10.6875, 10.6875}, 15}},
  });
  testLows2bfloat162Cases({
      {{{-0.3, -0.5}, {10.7, 3000000}}, {{-0.30078125, 10.6875}, 15}},
      {{{0.3, 0.5}, {-0.3, -0.5}}, {{0.30078125, -0.30078125}, 16}},
      {{{30, 50}, {0.3, 0.5}}, {{30, 0.30078125}, 14}},
      {{{0.432643, 0.23654}, {30, 50}}, {{0.43359375, 30}, 14}},
      {{{1, 5000}, {0.432643, 0.23654}}, {{1, 0.43359375}, 15}},
      {{{10.7, 3000000}, {1, 5000}}, {{10.6875, 1}, 15}},
  });
  testStcgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStcgCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStcsCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStcsCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStwbCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStwbCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStwtCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStwtCases({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
