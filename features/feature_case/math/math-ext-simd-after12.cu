// ===------------- math-ext-simd-after12.cu ------- *- CUDA -* -----------===//
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

template <typename T> void printInput(const string &FuncName, const T &Inputs) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = ";
}

template <typename T> void checkResult(const T &Expect, const T &result) {
  cout << result << " (expect " << Expect << ")";
  check(result == Expect);
}

template <typename T, typename V>
void checkResult(const T &Expect, const V &Expect1, const T &result,
                 const V &result1) {
  cout << result << " (expect " << Expect << ")";
  cout << " param1 = " << result1 << " (expect " << Expect1 << ")";
  check(result == Expect && result1 == Expect1);
}

template <typename T, typename V>
void checkResult(const T &Expect, const V &Expect1, const V &Expect2,
                 const T &result, const V &result1, const V &result2) {
  cout << result << " (expect " << Expect << ")";
  cout << " param1 = " << result1 << " (expect " << Expect1 << ")";
  cout << " param2 = " << result2 << " (expect " << Expect2 << ")";
  check(result == Expect && result1 == Expect1 && result2 == Expect2);
}

__global__ void viaddmax_s16x2(unsigned *const result, unsigned Input1,
                               unsigned Input2, unsigned Input3) {
  *result = __viaddmax_s16x2(Input1, Input2, Input3);
}

void testViaddmax_s16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                             TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_s16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmax_s16x2_relu(unsigned *const result, unsigned Input1,
                                    unsigned Input2, unsigned Input3) {
  *result = __viaddmax_s16x2_relu(Input1, Input2, Input3);
}

void testViaddmax_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                  TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmax_s32(int *const result, int Input1, int Input2,
                             int Input3) {
  *result = __viaddmax_s32(Input1, Input2, Input3);
}

void testViaddmax_s32Cases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_s32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmax_s32_relu(int *const result, int Input1, int Input2,
                                  int Input3) {
  *result = __viaddmax_s32_relu(Input1, Input2, Input3);
}

void testViaddmax_s32_reluCases(
    const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmax_u16x2(unsigned *const result, unsigned Input1,
                               unsigned Input2, unsigned Input3) {
  *result = __viaddmax_u16x2(Input1, Input2, Input3);
}

void testViaddmax_u16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                             TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_u16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmax_u32(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __viaddmax_u32(Input1, Input2, Input3);
}

void testViaddmax_u32Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmax_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmax_u32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_s16x2(unsigned *const result, unsigned Input1,
                               unsigned Input2, unsigned Input3) {
  *result = __viaddmin_s16x2(Input1, Input2, Input3);
}

void testViaddmin_s16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                             TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_s16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_s16x2_relu(unsigned *const result, unsigned Input1,
                                    unsigned Input2, unsigned Input3) {
  *result = __viaddmin_s16x2_relu(Input1, Input2, Input3);
}

void testViaddmin_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                  TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_s32(int *const result, int Input1, int Input2,
                             int Input3) {
  *result = __viaddmin_s32(Input1, Input2, Input3);
}

void testViaddmin_s32Cases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_s32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_s32_relu(int *const result, int Input1, int Input2,
                                  int Input3) {
  *result = __viaddmin_s32_relu(Input1, Input2, Input3);
}

void testViaddmin_s32_reluCases(
    const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_u16x2(unsigned *const result, unsigned Input1,
                               unsigned Input2, unsigned Input3) {
  *result = __viaddmin_u16x2(Input1, Input2, Input3);
}

void testViaddmin_u16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                             TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_u16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void viaddmin_u32(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __viaddmin_u32(Input1, Input2, Input3);
}

void testViaddmin_u32Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    viaddmin_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__viaddmin_u32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vibmax_s16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, bool *Input3, bool *Input4) {
  *result = __vibmax_s16x2(Input1, Input2, Input3, Input4);
}

void testVibmax_s16x2Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, pair<bool, bool>>>>
        &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  bool *result2;
  cudaMallocManaged(&result2, sizeof(*result2));
  for (const auto &TestCase : TestCases) {
    vibmax_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           result1, result2);
    cudaDeviceSynchronize();
    printInput("__vibmax_s16x2", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second.first,
                TestCase.second.second.second, *result, *result1, *result2);
  }
}

__global__ void vibmax_s32(int *const result, int Input1, int Input2,
                           bool *Input3) {
  *result = __vibmax_s32(Input1, Input2, Input3);
}

void testVibmax_s32Cases(
    const vector<pair<vector<int>, pair<int, bool>>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  for (const auto &TestCase : TestCases) {
    vibmax_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1], result1);
    cudaDeviceSynchronize();
    printInput("__vibmax_s32", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second, *result,
                *result1);
  }
}

__global__ void vibmax_u16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, bool *Input3, bool *Input4) {
  *result = __vibmax_u16x2(Input1, Input2, Input3, Input4);
}

void testVibmax_u16x2Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, pair<bool, bool>>>>
        &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  bool *result2;
  cudaMallocManaged(&result2, sizeof(*result2));
  for (const auto &TestCase : TestCases) {
    vibmax_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           result1, result2);
    cudaDeviceSynchronize();
    printInput("__vibmax_u16x2", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second.first,
                TestCase.second.second.second, *result, *result1, *result2);
  }
}

__global__ void vibmax_u32(unsigned *const result, unsigned Input1,
                           unsigned Input2, bool *Input3) {
  *result = __vibmax_u32(Input1, Input2, Input3);
}

void testVibmax_u32Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, bool>>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  for (const auto &TestCase : TestCases) {
    vibmax_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1], result1);
    cudaDeviceSynchronize();
    printInput("__vibmax_u32", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second, *result,
                *result1);
  }
}

__global__ void vibmin_s16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, bool *Input3, bool *Input4) {
  *result = __vibmin_s16x2(Input1, Input2, Input3, Input4);
}

void testVibmin_s16x2Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, pair<bool, bool>>>>
        &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  bool *result2;
  cudaMallocManaged(&result2, sizeof(*result2));
  for (const auto &TestCase : TestCases) {
    vibmin_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           result1, result2);
    cudaDeviceSynchronize();
    printInput("__vibmin_s16x2", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second.first,
                TestCase.second.second.second, *result, *result1, *result2);
  }
}

__global__ void vibmin_s32(int *const result, int Input1, int Input2,
                           bool *Input3) {
  *result = __vibmin_s32(Input1, Input2, Input3);
}

void testVibmin_s32Cases(
    const vector<pair<vector<int>, pair<int, bool>>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  for (const auto &TestCase : TestCases) {
    vibmin_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1], result1);
    cudaDeviceSynchronize();
    printInput("__vibmin_s32", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second, *result,
                *result1);
  }
}

__global__ void vibmin_u16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, bool *Input3, bool *Input4) {
  *result = __vibmin_u16x2(Input1, Input2, Input3, Input4);
}

void testVibmin_u16x2Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, pair<bool, bool>>>>
        &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  bool *result2;
  cudaMallocManaged(&result2, sizeof(*result2));
  for (const auto &TestCase : TestCases) {
    vibmin_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           result1, result2);
    cudaDeviceSynchronize();
    printInput("__vibmin_u16x2", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second.first,
                TestCase.second.second.second, *result, *result1, *result2);
  }
}

__global__ void vibmin_u32(unsigned *const result, unsigned Input1,
                           unsigned Input2, bool *Input3) {
  *result = __vibmin_u32(Input1, Input2, Input3);
}

void testVibmin_u32Cases(
    const vector<pair<vector<unsigned>, pair<unsigned, bool>>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  bool *result1;
  cudaMallocManaged(&result1, sizeof(*result1));
  for (const auto &TestCase : TestCases) {
    vibmin_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1], result1);
    cudaDeviceSynchronize();
    printInput("__vibmin_u32", TestCase.first);
    checkResult(TestCase.second.first, TestCase.second.second, *result,
                *result1);
  }
}

__global__ void vimax3_s16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __vimax3_s16x2(Input1, Input2, Input3);
}

void testVimax3_s16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_s16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax3_s16x2_relu(unsigned *const result, unsigned Input1,
                                  unsigned Input2, unsigned Input3) {
  *result = __vimax3_s16x2_relu(Input1, Input2, Input3);
}

void testVimax3_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax3_s32(int *const result, int Input1, int Input2,
                           int Input3) {
  *result = __vimax3_s32(Input1, Input2, Input3);
}

void testVimax3_s32Cases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                         TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_s32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax3_s32_relu(int *const result, int Input1, int Input2,
                                int Input3) {
  *result = __vimax3_s32_relu(Input1, Input2, Input3);
}

void testVimax3_s32_reluCases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                              TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax3_u16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __vimax3_u16x2(Input1, Input2, Input3);
}

void testVimax3_u16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_u16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax3_u32(unsigned *const result, unsigned Input1,
                           unsigned Input2, unsigned Input3) {
  *result = __vimax3_u32(Input1, Input2, Input3);
}

void testVimax3_u32Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax3_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                         TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimax3_u32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax_s16x2_relu(unsigned *const result, unsigned Input1,
                                 unsigned Input2) {
  *result = __vimax_s16x2_relu(Input1, Input2);
}

void testVimax_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1]);
    cudaDeviceSynchronize();
    printInput("__vimax_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimax_s32_relu(int *const result, int Input1, int Input2) {
  *result = __vimax_s32_relu(Input1, Input2);
}

void testVimax_s32_reluCases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimax_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1]);
    cudaDeviceSynchronize();
    printInput("__vimax_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_s16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __vimin3_s16x2(Input1, Input2, Input3);
}

void testVimin3_s16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_s16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_s16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_s16x2_relu(unsigned *const result, unsigned Input1,
                                  unsigned Input2, unsigned Input3) {
  *result = __vimin3_s16x2_relu(Input1, Input2, Input3);
}

void testVimin3_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                                TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_s32(int *const result, int Input1, int Input2,
                           int Input3) {
  *result = __vimin3_s32(Input1, Input2, Input3);
}

void testVimin3_s32Cases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_s32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                         TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_s32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_s32_relu(int *const result, int Input1, int Input2,
                                int Input3) {
  *result = __vimin3_s32_relu(Input1, Input2, Input3);
}

void testVimin3_s32_reluCases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                              TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_u16x2(unsigned *const result, unsigned Input1,
                             unsigned Input2, unsigned Input3) {
  *result = __vimin3_u16x2(Input1, Input2, Input3);
}

void testVimin3_u16x2Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_u16x2<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                           TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_u16x2", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin3_u32(unsigned *const result, unsigned Input1,
                           unsigned Input2, unsigned Input3) {
  *result = __vimin3_u32(Input1, Input2, Input3);
}

void testVimin3_u32Cases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin3_u32<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1],
                         TestCase.first[2]);
    cudaDeviceSynchronize();
    printInput("__vimin3_u32", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin_s16x2_relu(unsigned *const result, unsigned Input1,
                                 unsigned Input2) {
  *result = __vimin_s16x2_relu(Input1, Input2);
}

void testVimin_s16x2_reluCases(
    const vector<pair<vector<unsigned>, unsigned>> &TestCases) {
  unsigned *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin_s16x2_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1]);
    cudaDeviceSynchronize();
    printInput("__vimin_s16x2_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

__global__ void vimin_s32_relu(int *const result, int Input1, int Input2) {
  *result = __vimin_s32_relu(Input1, Input2);
}

void testVimin_s32_reluCases(const vector<pair<vector<int>, int>> &TestCases) {
  int *result;
  cudaMallocManaged(&result, sizeof(*result));
  for (const auto &TestCase : TestCases) {
    vimin_s32_relu<<<1, 1>>>(result, TestCase.first[0], TestCase.first[1]);
    cudaDeviceSynchronize();
    printInput("__vimin_s32_relu", TestCase.first);
    checkResult(TestCase.second, *result);
  }
}

int main() {
  testViaddmax_s16x2Cases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, 4294967295}, 4294919472},
      {{4294967295, 2147483647, 214321}, 2147370289},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testViaddmax_s16x2_reluCases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, 4294967295}, 17712},
      {{4294967295, 2147483647, 214321}, 2147370289},
      {{4294967295, 4294967295, 4294967295}, 0},
      {{3, 4, 8}, 8},
  });
  testViaddmax_s32Cases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, -1}, -1},
      {{-1, 2147483647, 214321}, 2147483646},
      {{-2147483648, -2147483648, -2147483648}, 0},
      {{3, 4, 8}, 8},
  });
  testViaddmax_s32_reluCases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, -1}, 0},
      {{-1, 2147483647, 214321}, 2147483646},
      {{-2147483648, -2147483648, -2147483648}, 0},
      {{3, 4, 8}, 8},
  });
  testViaddmax_u16x2Cases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, 4294967295}, 4294967295},
      {{4294967295, 2147483647, 214321}, 2147418110},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testViaddmax_u32Cases({
      {{4, 3, 2}, 7},
      {{214321, 2147483647, 4294967295}, 4294967295},
      {{4294967295, 2147483647, 214321}, 2147483646},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testViaddmin_s16x2Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 2147680255},
      {{4294967295, 2147483647, 214321}, 262142},
      {{4294967295, 4294967295, 4294967295}, 4294901758},
      {{3, 4, 8}, 7},
  });
  testViaddmin_s16x2_reluCases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 0},
      {{4294967295, 2147483647, 214321}, 196608},
      {{4294967295, 4294967295, 4294967295}, 0},
      {{3, 4, 8}, 7},
  });
  testViaddmin_s32Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, -1}, -2147269328},
      {{-1, 2147483647, 214321}, 214321},
      {{-2147483648, -2147483648, -2147483648}, -2147483648},
      {{3, 4, 8}, 7},
  });
  testViaddmin_s32_reluCases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, -1}, 0},
      {{-1, 2147483647, 214321}, 214321},
      {{-2147483648, -2147483648, -2147483648}, 0},
      {{3, 4, 8}, 7},
  });
  testViaddmin_u16x2Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 2147632432},
      {{4294967295, 2147483647, 214321}, 214321},
      {{4294967295, 4294967295, 4294967295}, 4294901758},
      {{3, 4, 8}, 7},
  });
  testViaddmin_u32Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 2147697968},
      {{4294967295, 2147483647, 214321}, 214321},
      {{4294967295, 4294967295, 4294967295}, 4294967294},
      {{3, 4, 8}, 7},
  });
  testVibmax_s16x2Cases({
      {{4, 3}, {4, {true, true}}},
      {{214321, 2147483647}, {2147435825, {false, true}}},
      {{4294967295, 2147483647}, {2147483647, {false, true}}},
      {{4294967295, 4294967295}, {4294967295, {true, true}}},
      {{3, 4}, {4, {true, false}}},
  });
  testVibmax_s32Cases({
      {{4, 3}, {4, true}},
      {{214321, 2147483647}, {2147483647, false}},
      {{-1, 2147483647}, {2147483647, false}},
      {{-1, -1}, {-1, true}},
      {{3, 4}, {4, false}},
  });
  testVibmax_u16x2Cases({
      {{4, 3}, {4, {true, true}}},
      {{214321, 2147483647}, {2147483647, {false, false}}},
      {{4294967295, 2147483647}, {4294967295, {true, true}}},
      {{4294967295, 4294967295}, {4294967295, {true, true}}},
      {{3, 4}, {4, {true, false}}},
  });
  testVibmax_u32Cases({
      {{4, 3}, {4, true}},
      {{214321, 2147483647}, {2147483647, false}},
      {{4294967295, 2147483647}, {4294967295, true}},
      {{4294967295, 4294967295}, {4294967295, true}},
      {{3, 4}, {4, false}},
  });
  testVibmin_s16x2Cases({
      {{4, 3}, {3, {true, false}}},
      {{214321, 2147483647}, {262143, {true, false}}},
      {{4294967295, 2147483647}, {4294967295, {true, true}}},
      {{4294967295, 4294967295}, {4294967295, {true, true}}},
      {{3, 4}, {3, {true, true}}},
  });
  testVibmin_s32Cases({
      {{4, 3}, {3, false}},
      {{214321, 2147483647}, {214321, true}},
      {{-1, 2147483647}, {-1, true}},
      {{-1, -1}, {-1, true}},
      {{3, 4}, {3, true}},
  });
  testVibmin_u16x2Cases({
      {{4, 3}, {3, {true, false}}},
      {{214321, 2147483647}, {214321, {true, true}}},
      {{4294967295, 2147483647}, {2147483647, {false, true}}},
      {{4294967295, 4294967295}, {4294967295, {true, true}}},
      {{3, 4}, {3, {true, true}}},
  });
  testVibmin_u32Cases({
      {{4, 3}, {3, false}},
      {{214321, 2147483647}, {214321, true}},
      {{4294967295, 2147483647}, {2147483647, false}},
      {{4294967295, 4294967295}, {4294967295, true}},
      {{3, 4}, {3, true}},
  });
  testVimax3_s16x2Cases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, 4294967295}, 2147435825},
      {{4294967295, 2147483647, 214321}, 2147435825},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testVimax3_s16x2_reluCases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, 4294967295}, 2147435825},
      {{4294967295, 2147483647, 214321}, 2147435825},
      {{4294967295, 4294967295, 4294967295}, 0},
      {{3, 4, 8}, 8},
  });
  testVimax3_s32Cases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, -1}, 2147483647},
      {{-1, 2147483647, 214321}, 2147483647},
      {{-2147483648, -2147483648, -2147483648}, -2147483648},
      {{3, 4, 8}, 8},
  });
  testVimax3_s32_reluCases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, -1}, 2147483647},
      {{-1, 2147483647, 214321}, 2147483647},
      {{-2147483648, -2147483648, -2147483648}, 0},
      {{3, 4, 8}, 8},
  });
  testVimax3_u16x2Cases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, 4294967295}, 4294967295},
      {{4294967295, 2147483647, 214321}, 4294967295},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testVimax3_u32Cases({
      {{4, 3, 2}, 4},
      {{214321, 2147483647, 4294967295}, 4294967295},
      {{4294967295, 2147483647, 214321}, 4294967295},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 8},
  });
  testVimax_s16x2_reluCases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147435825},
      {{4294967295, 2147483647}, 2147418112},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 4},
  });
  testVimax_s32_reluCases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147483647},
      {{-1, 2147483647}, 2147483647},
      {{-2147483648, -2147483648}, 0},
      {{3, 4}, 4},
  });
  testVimin3_s16x2Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 4294967295},
      {{4294967295, 2147483647, 214321}, 4294967295},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 3},
  });
  testVimin3_s16x2_reluCases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 0},
      {{4294967295, 2147483647, 214321}, 0},
      {{4294967295, 4294967295, 4294967295}, 0},
      {{3, 4, 8}, 3},
  });
  testVimin3_s32Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, -1}, -1},
      {{-1, 2147483647, 214321}, -1},
      {{-2147483648, -2147483648, -2147483648}, -2147483648},
      {{3, 4, 8}, 3},
  });
  testVimin3_s32_reluCases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, -1}, 0},
      {{-1, 2147483647, 214321}, 0},
      {{-2147483648, -2147483648, -2147483648}, 0},
      {{3, 4, 8}, 3},
  });
  testVimin3_u16x2Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 214321},
      {{4294967295, 2147483647, 214321}, 214321},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 3},
  });
  testVimin3_u32Cases({
      {{4, 3, 2}, 2},
      {{214321, 2147483647, 4294967295}, 214321},
      {{4294967295, 2147483647, 214321}, 214321},
      {{4294967295, 4294967295, 4294967295}, 4294967295},
      {{3, 4, 8}, 3},
  });
  testVimin_s16x2_reluCases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 196608},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 3},
  });
  testVimin_s32_reluCases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 214321},
      {{-1, 2147483647}, 0},
      {{-2147483648, -2147483648}, 0},
      {{3, 4}, 3},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
