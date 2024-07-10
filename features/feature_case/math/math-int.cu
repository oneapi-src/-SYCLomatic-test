// ===--------------- math-int.cu ---------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>
#include <vector>

using namespace std;

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

template <typename T> void printFunc(const string &FuncName, const T &Input) {
  cout << FuncName << "(" << Input << ") ";
}

template <typename T1, typename T2>
void printFunc(const string &FuncName, const pair<T1, T2> &Input) {
  cout << FuncName << "(" << Input.first << ", " << Input.second << ")";
}

template <typename T> void checkResult(const T &Expect, const T &Result) {
  cout << " = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

// Integer Mathematical Functions

__global__ void _llmax(long long *const Result, long long Input1,
                       long long Input2) {
  *Result = llmax(Input1, Input2);
}

void testLlmaxCases(
    const vector<pair<pair<long long, long long>, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _llmax<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("llmax", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

__global__ void _llmin(long long *const Result, long long Input1,
                       long long Input2) {
  *Result = llmin(Input1, Input2);
}

void testLlminCases(
    const vector<pair<pair<long long, long long>, long long>> &TestCases) {
  long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _llmin<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("llmin", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

__global__ void _ullmax(unsigned long long *const Result,
                        unsigned long long Input1, unsigned long long Input2) {
  *Result = ullmax(Input1, Input2);
}

void testUllmaxCases(
    const vector<pair<pair<unsigned long long, unsigned long long>,
                      unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _ullmax<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("ullmax", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

__global__ void _ullmin(unsigned long long *const Result,
                        unsigned long long Input1, unsigned long long Input2) {
  *Result = ullmin(Input1, Input2);
}

void testUllminCases(
    const vector<pair<pair<unsigned long long, unsigned long long>,
                      unsigned long long>> &TestCases) {
  unsigned long long *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _ullmin<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("ullmin", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

__global__ void _umax(unsigned *const Result, unsigned Input1,
                      unsigned Input2) {
  *Result = umax(Input1, Input2);
}

void testUmaxCases(
    const vector<pair<pair<unsigned, unsigned>, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _umax<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("umax", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

__global__ void _umin(unsigned *const Result, unsigned Input1,
                      unsigned Input2) {
  *Result = umin(Input1, Input2);
}

void testUminCases(
    const vector<pair<pair<unsigned, unsigned>, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _umin<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    printFunc("umin", TestCase.first);
    checkResult(TestCase.second, *Result);
  }
}

int main() {
  testLlmaxCases({
      {{1, 2}, 2},
      {{-1, -2}, -1},
      {{1, -2}, 1},
      {{-1, 2}, 2},
      {{45212221678, 221332142421}, 221332142421},
  });
  testLlminCases({
      {{1, 2}, 1},
      {{-1, -2}, -2},
      {{1, -2}, -2},
      {{-1, 2}, -1},
      {{45212221678, 221332142421}, 45212221678},
  });
  testUllmaxCases({
      {{1, 2}, 2},
      {{18446744073709551615, 18446744073709551614}, 18446744073709551615},
      {{1, 18446744073709551614}, 18446744073709551614},
      {{18446744073709551615, 2}, 18446744073709551615},
      {{45212221678, 221332142421}, 221332142421},
  });
  testUllminCases({
      {{1, 2}, 1},
      {{18446744073709551615, 18446744073709551614}, 18446744073709551614},
      {{1, 18446744073709551614}, 1},
      {{18446744073709551615, 2}, 2},
      {{45212221678, 221332142421}, 45212221678},
  });
  testUmaxCases({
      {{1, 2}, 2},
      {{4294967295, 4294967294}, 4294967295},
      {{1, 4294967294}, 4294967294},
      {{4294967295, 2}, 4294967295},
      {{2262548718, 2288810325}, 2288810325},
  });
  testUminCases({
      {{1, 2}, 1},
      {{4294967295, 4294967294}, 4294967294},
      {{1, 4294967294}, 1},
      {{4294967295, 2}, 2},
      {{2262548718, 2288810325}, 2262548718},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
