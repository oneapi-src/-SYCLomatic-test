// ===-------------- math-half-conv.cu -------=--------- *- CUDA -* -------===//
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
    std::cout << " ---- passed" << std::endl;
    passed++;
  } else {
    std::cout << " ---- failed" << std::endl;
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
                 const __half &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__half2float(it));
  }
  float FExpect = __half2float(Expect);
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  float2 FExpect{__half2float(Expect.x), __half2float(Expect.y)};
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__half2> &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__half2float(it.x), __half2float(it.y)});
  }
  float2 FExpect{__half2float(Expect.x), __half2float(Expect.y)};
  checkResult(FuncName, FInputs, FExpect, Result, precision);
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

void checkResult(const string &FuncName, const vector<__half2> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__half2float(it.x), __half2float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result, precision);
}

__global__ void setValue(__half *Input1, const __half Input2) {
  *Input1 = Input2;
}

__global__ void setValue(__half2 *Input1, const __half2 Input2) {
  *Input1 = Input2;
}

__global__ void double2half(float *const Result, double Input1) {
  *Result = __double2half(Input1);
}

void testDouble2halfCases(const vector<pair<double, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    double2half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__double2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __double2half(TestCase.first);
    checkResult("(host)__double2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

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
    checkResult("__float2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2half(TestCase.first);
    checkResult("(host)__float2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
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
    checkResult("__float2half_rd", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2half_rd(TestCase.first);
    checkResult("(host)__float2half_rd", {TestCase.first},
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
    checkResult("__float2half_rn", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2half_rn(TestCase.first);
    checkResult("(host)__float2half_rn", {TestCase.first},
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
    checkResult("__float2half_ru", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2half_ru(TestCase.first);
    checkResult("(host)__float2half_ru", {TestCase.first},
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
    checkResult("__float2half_rz", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2half_rz(TestCase.first);
    checkResult("(host)__float2half_rz", {TestCase.first},
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

__global__ void half2half2(float *const Result, __half Input1) {
  auto ret = __half2half2(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testHalf2half2Cases(const vector<pair<__half, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    half2half2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half2half2", vector<float>{TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
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

__global__ void half_as_short(short *const Result, __half Input1) {
  *Result = __half_as_short(Input1);
}

void testHalf_as_shortCases(const vector<pair<__half, short>> &TestCases) {
  short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half_as_short<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half_as_short", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void half_as_ushort(unsigned short *const Result, __half Input1) {
  *Result = __half_as_ushort(Input1);
}

void testHalf_as_ushortCases(
    const vector<pair<__half, unsigned short>> &TestCases) {
  unsigned short *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half_as_ushort<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__half_as_ushort", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void halves2half2(float *const Result, __half Input1,
                             __half Input2) {
  auto ret = __halves2half2(Input1, Input2);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testHalves2half2Cases(
    const vector<pair<pair<__half, __half>, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    halves2half2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__halves2half2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void high2float(float *const Result, __half2 Input1) {
  *Result = __high2float(Input1);
}

void testHigh2floatCases(const vector<pair<__half2, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    high2float<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__high2float", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void high2half(float *const Result, __half2 Input1) {
  *Result = __high2half(Input1);
}

void testHigh2halfCases(const vector<pair<__half2, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    high2half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__high2half", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void high2half2(float *const Result, __half2 Input1) {
  auto ret = __high2half2(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testHigh2half2Cases(const vector<pair<__half2, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    high2half2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__high2half2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void highs2half2(float *const Result, __half2 Input1,
                            __half2 Input2) {
  auto ret = __highs2half2(Input1, Input2);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testHighs2half2Cases(
    const vector<pair<pair<__half2, __half2>, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    highs2half2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__highs2half2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
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

__global__ void ldca(float *const Result, __half *Input1) {
  *Result = __ldca(Input1);
}

void testLdcaCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldca(float *const Result, __half2 *Input1) {
  auto ret = __ldca(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcaCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __half *Input1) {
  *Result = __ldcg(Input1);
}

void testLdcgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __half2 *Input1) {
  auto ret = __ldcg(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __half *Input1) {
  *Result = __ldcs(Input1);
}

void testLdcsCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __half2 *Input1) {
  auto ret = __ldcs(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcsCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __half *Input1) {
  *Result = __ldcv(Input1);
}

void testLdcvCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __half2 *Input1) {
  auto ret = __ldcv(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcvCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldg(float *const Result, __half *Input1) {
  *Result = __ldg(Input1);
}

void testLdgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldg(float *const Result, __half2 *Input1) {
  auto ret = __ldg(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __half *Input1) {
  *Result = __ldlu(Input1);
}

void testLdluCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __half2 *Input1) {
  auto ret = __ldlu(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdluCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
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

__global__ void low2float(float *const Result, __half2 Input1) {
  *Result = __low2float(Input1);
}

void testLow2floatCases(const vector<pair<__half2, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    low2float<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__low2float", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void low2half(float *const Result, __half2 Input1) {
  *Result = __low2half(Input1);
}

void testLow2halfCases(const vector<pair<__half2, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    low2half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__low2half", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void low2half2(float *const Result, __half2 Input1) {
  auto ret = __low2half2(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLow2half2Cases(const vector<pair<__half2, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    low2half2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__low2half2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void lowhigh2highlow(float *const Result, __half2 Input1) {
  auto ret = __lowhigh2highlow(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLowhigh2highlowCases(
    const vector<pair<__half2, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    lowhigh2highlow<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__lowhigh2highlow", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void lows2half2(float *const Result, __half2 Input1,
                           __half2 Input2) {
  auto ret = __lows2half2(Input1, Input2);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLows2half2Cases(
    const vector<pair<pair<__half2, __half2>, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    lows2half2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__lows2half2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
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

__global__ void short_as_half(float *const Result, short Input1) {
  *Result = __short_as_half(Input1);
}

void testShort_as_halfCases(const vector<pair<short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    short_as_half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__short_as_half", vector<int>{TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void stcg(float *const Result, __half Input1, __half *const Temp) {
  __stcg(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStcgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcg(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stcg(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStcgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stcs(float *const Result, __half Input1, __half *const Temp) {
  __stcs(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStcsCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcs(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stcs(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStcsCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwb(float *const Result, __half Input1, __half *const Temp) {
  __stwb(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStwbCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwb(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stwb(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStwbCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwt(float *const Result, __half Input1, __half *const Temp) {
  __stwt(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStwtCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwt(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stwt(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStwtCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
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

__global__ void ushort_as_half(float *const Result, unsigned short Input1) {
  *Result = __ushort_as_half(Input1);
}

void testUshort_as_halfCases(
    const vector<pair<unsigned short, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    ushort_as_half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__ushort_as_half", vector<int>{(int)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void make_half2(float *const Result, __half Input1, __half Input2) {
  auto ret = make_half2(Input1, Input2);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testMake_half2Cases(
    const vector<pair<pair<__half, __half>, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result) * 2);
  for (const auto &TestCase : TestCases) {
    make_half2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__make_half2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

int main() {
  testDouble2halfCases({
      {-0.3, {-0.3, 4}},
      {0.3, {0.3, 4}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
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
  testHalf2half2Cases({
      {-0.3, {{-0.3000488281250000, -0.3000488281250000}, 16}},
      {0.3, {{0.3000488281250000, 0.3000488281250000}, 16}},
      {0.5, {{0.5, 0.5}, 16}},
      {23, {{23, 23}, 14}},
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
      // {-0.3, 0},
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
      // {-0.3, 0},
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
      // {-0.3, 0},
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
  testHalf_as_shortCases({
      {-0.3, -19251},
      {0.3, 13517},
      {30, 20352},
      {0.432643, 14060},
      {1, 15360},
      {10.7, 18778},
  });
  testHalf_as_ushortCases({
      {-0.3, 46285},
      {0.3, 13517},
      {30, 20352},
      {0.432643, 14060},
      {1, 15360},
      {10.7, 18778},
  });
  testHalves2half2Cases({
      {{-0.3, -0.4}, {{-0.300048828125, -0.39990234375}, 16}},
      {{0, 0.7}, {{0, 0.7001953125}, 16}},
      {{1, 100.6}, {{1, 100.625}, 14}},
      {{100.6, 1}, {{100.625, 1}, 14}},
  });
  testHigh2floatCases({
      {{-0.3, -0.4}, {-0.39990234375, 16}},
      {{0, 0.7}, {0.7001953125, 16}},
      {{1, 100.6}, {100.625, 14}},
      {{100.6, 1}, {1, 15}},
  });
  testHigh2halfCases({
      {{-0.3, -0.4}, {-0.39990234375, 16}},
      {{0, 0.7}, {0.7001953125, 16}},
      {{1, 100.6}, {100.625, 14}},
      {{100.6, 1}, {1, 15}},
  });
  testHigh2half2Cases({
      {{-0.3, -0.4}, {{-0.39990234375, -0.39990234375}, 16}},
      {{0, 0.7}, {{0.7001953125, 0.7001953125}, 16}},
      {{1, 100.6}, {{100.625, 100.625}, 14}},
      {{100.6, 1}, {{1, 1}, 15}},
  });
  testHighs2half2Cases({
      {{{-0.3, -0.4}, {0, 0.7}}, {{-0.39990234375, 0.7001953125}, 16}},
      {{{0, 0.7}, {1, 100.6}}, {{0.7001953125, 100.625}, 14}},
      {{{1, 100.6}, {100.6, 1}}, {{100.625, 1}, 14}},
      {{{100.6, 1}, {-0.3, -0.4}}, {{1, -0.39990234375}, 15}},
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
  testLdcaCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcaCases2({
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
  testLdcgCases2({
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
  testLdcsCases2({
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
  testLdcvCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdgCases2({
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
  testLdluCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
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
  testLow2floatCases({
      {{-0.3, -0.4}, {-0.300048828125, 16}},
      {{0, 0.7}, {0, 37}},
      {{1, 100.6}, {1, 15}},
      {{100.6, 1}, {100.625, 14}},
  });
  testLow2halfCases({
      {{-0.3, -0.4}, {-0.300048828125, 16}},
      {{0, 0.7}, {0, 37}},
      {{1, 100.6}, {1, 15}},
      {{100.6, 1}, {100.625, 14}},
  });
  testLow2half2Cases({
      {{-0.3, -0.4}, {{-0.300048828125, -0.300048828125}, 16}},
      {{0, 0.7}, {{0, 0}, 37}},
      {{1, 100.6}, {{1, 1}, 15}},
      {{100.6, 1}, {{100.625, 100.625}, 14}},
  });
  testLowhigh2highlowCases({
      {{-0.3, -0.4}, {{-0.39990234375, -0.300048828125}, 16}},
      {{0, 0.7}, {{0.7001953125, 0}, 16}},
      {{1, 100.6}, {{100.625, 1}, 14}},
      {{100.6, 1}, {{1, 100.625}, 14}},
  });
  testLows2half2Cases({
      {{{-0.3, -0.4}, {0, 0.7}}, {{-0.300048828125, 0}, 16}},
      {{{0, 0.7}, {1, 100.6}}, {{0, 1}, 15}},
      {{{1, 100.6}, {100.6, 1}}, {{1, 100.625}, 14}},
      {{{100.6, 1}, {-0.3, -0.4}}, {{100.625, -0.300048828125}, 14}},
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
  testShort_as_halfCases({
      {-10000, {-158, 13}},
      {10000, {0.027587890625, 17}},
      {30000, {21248, 11}},
      {3000, {0.0002355575561523438, 19}},
      {1000, {0.0000596046447753906, 19}},
      {-3000, {-17536, 11}},
  });
  testStcgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStcgCases2({
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
  testStcsCases2({
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
  testStwbCases2({
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
  testStwtCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
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
      {10001, {10000, 12}},
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
  testUshort_as_halfCases({
      {55536, {-158, 13}},
      {10000, {0.027587890625, 17}},
      {30000, {21248, 11}},
      {3000, {0.0002355575561523438, 19}},
      {1000, {0.0000596046447753906, 19}},
      {62536, {-17536, 11}},
  });
  testMake_half2Cases({
      {{-0.3, -0.4}, {{-0.300048828125, -0.39990234375}, 16}},
      {{0, 0.7}, {{0, 0.7001953125}, 16}},
      {{1, 100.6}, {{1, 100.625}, 14}},
      {{100.6, 1}, {{100.625, 1}, 14}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
