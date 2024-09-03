// ====------------ math-emu-simd.cu---------- *- CUDA -* ---------------===////
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

typedef pair<unsigned int, unsigned int> Uint_pair;

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

void checkResult(const string &FuncName, const vector<unsigned int> &Inputs,
                 const unsigned int &Expect, const unsigned int &DeviceResult) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << DeviceResult << " (expect " << Expect << ")";
  check(DeviceResult == Expect);
}

__global__ void vabs2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vabs2(Input1);
}

void testVabs2Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabs2<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vabs2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vabs4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vabs4(Input1);
}

void testVabs4Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabs4<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vabs4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vabsdiffs2(unsigned int *const DeviceResult,
                           unsigned int Input1, unsigned int Input2) {
  *DeviceResult = __vabsdiffs2(Input1, Input2);
}

void testVabsdiffs2Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsdiffs2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                         TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vabsdiffs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vabsdiffs4(unsigned int *const DeviceResult,
                           unsigned int Input1, unsigned int Input2) {
  *DeviceResult = __vabsdiffs4(Input1, Input2);
}

void testVabsdiffs4Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsdiffs4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                         TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vabsdiffs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vabsdiffu2(unsigned int *const DeviceResult,
                           unsigned int Input1, unsigned int Input2) {
  *DeviceResult = __vabsdiffu2(Input1, Input2);
}

void testVabsdiffu2Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsdiffu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                         TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vabsdiffu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vabsdiffu4(unsigned int *const DeviceResult,
                           unsigned int Input1, unsigned int Input2) {
  *DeviceResult = __vabsdiffu4(Input1, Input2);
}

void testVabsdiffu4Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsdiffu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                         TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vabsdiffu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vabsss2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vabsss2(Input1);
}

void testVabsss2Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsss2<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vabsss2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vabsss4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vabsss4(Input1);
}

void testVabsss4Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vabsss4<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vabsss4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vadd2(unsigned int *const DeviceResult, unsigned int Input1,
                      unsigned int Input2) {
  *DeviceResult = __vadd2(Input1, Input2);
}

void testVadd2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vadd2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vadd2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vadd4(unsigned int *const DeviceResult, unsigned int Input1,
                      unsigned int Input2) {
  *DeviceResult = __vadd4(Input1, Input2);
}

void testVadd4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vadd4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vadd4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vaddss2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vaddss2(Input1, Input2);
}

void testVaddss2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vaddss2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vaddss2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vaddss4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vaddss4(Input1, Input2);
}

void testVaddss4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vaddss4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vaddss4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vaddus2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vaddus2(Input1, Input2);
}

void testVaddus2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vaddus2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vaddus2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vaddus4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vaddus4(Input1, Input2);
}

void testVaddus4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vaddus4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vaddus4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vavgs2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vavgs2(Input1, Input2);
}

void testVavgs2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vavgs2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vavgs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vavgs4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vavgs4(Input1, Input2);
}

void testVavgs4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vavgs4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vavgs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vavgu2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vavgu2(Input1, Input2);
}

void testVavgu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vavgu2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vavgu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vavgu4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vavgu4(Input1, Input2);
}

void testVavgu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vavgu4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vavgu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpeq2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vcmpeq2(Input1, Input2);
}

void testVcmpeq2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpeq2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpeq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpeq4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vcmpeq4(Input1, Input2);
}

void testVcmpeq4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpeq4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpeq4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpges2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpges2(Input1, Input2);
}

void testVcmpges2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpges2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpges2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpges4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpges4(Input1, Input2);
}

void testVcmpges4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpges4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpges4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgeu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgeu2(Input1, Input2);
}

void testVcmpgeu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgeu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgeu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgeu4(Input1, Input2);
}

void testVcmpgeu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgeu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgeu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgts2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgts2(Input1, Input2);
}

void testVcmpgts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgts2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgts4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgts4(Input1, Input2);
}

void testVcmpgts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgts4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgtu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgtu2(Input1, Input2);
}

void testVcmpgtu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgtu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpgtu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpgtu4(Input1, Input2);
}

void testVcmpgtu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpgtu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpgtu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmples2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmples2(Input1, Input2);
}

void testVcmples2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmples2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmples2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmples4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmples4(Input1, Input2);
}

void testVcmples4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmples4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmples4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpleu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpleu2(Input1, Input2);
}

void testVcmpleu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpleu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpleu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpleu4(Input1, Input2);
}

void testVcmpleu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpleu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpleu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmplts2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmplts2(Input1, Input2);
}

void testVcmplts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmplts2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmplts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmplts4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmplts4(Input1, Input2);
}

void testVcmplts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmplts4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmplts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpltu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpltu2(Input1, Input2);
}

void testVcmpltu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpltu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpltu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vcmpltu4(Input1, Input2);
}

void testVcmpltu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpltu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpltu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpne2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vcmpne2(Input1, Input2);
}

void testVcmpne2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpne2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vcmpne4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vcmpne4(Input1, Input2);
}

void testVcmpne4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vcmpne4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vcmpne4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vhaddu2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vhaddu2(Input1, Input2);
}

void testVhaddu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vhaddu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vhaddu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vhaddu4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vhaddu4(Input1, Input2);
}

void testVhaddu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vhaddu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vhaddu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmaxs2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmaxs2(Input1, Input2);
}

void testVmaxs2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmaxs2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmaxs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmaxs4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmaxs4(Input1, Input2);
}

void testVmaxs4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmaxs4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmaxs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmaxu2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmaxu2(Input1, Input2);
}

void testVmaxu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmaxu2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmaxu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmaxu4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmaxu4(Input1, Input2);
}

void testVmaxu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmaxu4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmaxu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmins2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmins2(Input1, Input2);
}

void testVmins2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmins2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmins2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vmins4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vmins4(Input1, Input2);
}

void testVmins4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vmins4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vmins4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vminu2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vminu2(Input1, Input2);
}

void testVminu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vminu2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vminu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vminu4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vminu4(Input1, Input2);
}

void testVminu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vminu4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vminu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vneg2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vneg2(Input1);
}

void testVneg2Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vneg2<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vneg2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vneg4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vneg4(Input1);
}

void testVneg4Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vneg4<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vneg4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vnegss2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vnegss2(Input1);
}

void testVnegss2Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vnegss2<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vnegss2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vnegss4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = __vnegss4(Input1);
}

void testVnegss4Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vnegss4<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__vnegss4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

__global__ void vsads2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vsads2(Input1, Input2);
}

void testVsads2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsads2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsads2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsads4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vsads4(Input1, Input2);
}

void testVsads4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsads4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsads4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsadu2(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vsadu2(Input1, Input2);
}

void testVsadu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsadu2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsadu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsadu4(unsigned int *const DeviceResult, unsigned int Input1,
                       unsigned int Input2) {
  *DeviceResult = __vsadu4(Input1, Input2);
}

void testVsadu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsadu4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsadu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vseteq2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vseteq2(Input1, Input2);
}

void testVseteq2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vseteq2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vseteq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vseteq4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vseteq4(Input1, Input2);
}

void testVseteq4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vseteq4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vseteq4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetges2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetges2(Input1, Input2);
}

void testVsetges2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetges2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetges2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetges4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetges4(Input1, Input2);
}

void testVsetges4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetges4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetges4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgeu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgeu2(Input1, Input2);
}

void testVsetgeu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgeu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgeu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgeu4(Input1, Input2);
}

void testVsetgeu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgeu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgeu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgts2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgts2(Input1, Input2);
}

void testVsetgts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgts2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgts4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgts4(Input1, Input2);
}

void testVsetgts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgts4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgtu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgtu2(Input1, Input2);
}

void testVsetgtu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgtu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetgtu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetgtu4(Input1, Input2);
}

void testVsetgtu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetgtu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetgtu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetles2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetles2(Input1, Input2);
}

void testVsetles2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetles2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetles2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetles4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetles4(Input1, Input2);
}

void testVsetles4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetles4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetles4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetleu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetleu2(Input1, Input2);
}

void testVsetleu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetleu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetleu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetleu4(Input1, Input2);
}

void testVsetleu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetleu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetleu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetlts2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetlts2(Input1, Input2);
}

void testVsetlts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetlts2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetlts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetlts4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetlts4(Input1, Input2);
}

void testVsetlts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetlts4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetlts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetltu2(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetltu2(Input1, Input2);
}

void testVsetltu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetltu2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetltu4(unsigned int *const DeviceResult, unsigned int Input1,
                         unsigned int Input2) {
  *DeviceResult = __vsetltu4(Input1, Input2);
}

void testVsetltu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetltu4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                       TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetltu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetne2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsetne2(Input1, Input2);
}

void testVsetne2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetne2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsetne4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsetne4(Input1, Input2);
}

void testVsetne4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsetne4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsetne4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsub2(unsigned int *const DeviceResult, unsigned int Input1,
                      unsigned int Input2) {
  *DeviceResult = __vsub2(Input1, Input2);
}

void testVsub2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsub2<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsub2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsub4(unsigned int *const DeviceResult, unsigned int Input1,
                      unsigned int Input2) {
  *DeviceResult = __vsub4(Input1, Input2);
}

void testVsub4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsub4<<<1, 1>>>(DeviceResult, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsub4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsubss2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsubss2(Input1, Input2);
}

void testVsubss2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsubss2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsubss2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsubss4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsubss4(Input1, Input2);
}

void testVsubss4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsubss4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsubss4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsubus2(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsubus2(Input1, Input2);
}

void testVsubus2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsubus2<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsubus2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

__global__ void vsubus4(unsigned int *const DeviceResult, unsigned int Input1,
                        unsigned int Input2) {
  *DeviceResult = __vsubus4(Input1, Input2);
}

void testVsubus4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  unsigned int *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    vsubus4<<<1, 1>>>(DeviceResult, TestCase.first.first,
                      TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__vsubus4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

int main() {
  testVabs2Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2147418113}, // 7FFF,FFFF-->7FFF,0001
      {0, 0},
      {4294967295, 65537}, // FFFF,FFFF-->0001,0001
  });
  testVabs4Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2130772225}, // 7F,FF,FF,FF-->7F,01,01,01
      {0, 0},
      {4294967295, 16843009}, // FF,FF,FF,FF-->01,01,01,01
  });
  testVabsdiffs2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147239218},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffs4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2130986546},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147269326},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147269326},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsss2Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2147418113},
      {0, 0},
      {4294967295, 65537},
  });
  testVabsss4Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2130772225},
      {0, 0},
      {4294967295, 16843009},
  });
  testVadd2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147632432},
      {{4294967295, 2147483647}, 2147418110},
      {{4294967295, 4294967295}, 4294901758},
      {{3, 4}, 7},
  });
  testVadd4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2130854960},
      {{4294967295, 2147483647}, 2130640638},
      {{4294967295, 4294967295}, 4278124286},
      {{3, 4}, 7},
  });
  testVaddss2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147435824}, // 3,4531+7FFF,FFFF-->7FFF,4530
      {{4294967295, 2147483647}, 2147418110},
      {{4294967295, 4294967295}, 4294901758},
      {{3, 4}, 7},
  });
  testVaddss4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2130854960},
      {{4294967295, 2147483647}, 2130640638},
      {{4294967295, 4294967295}, 4278124286},
      {{3, 4}, 7},
  });
  testVaddus2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147680255},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 7},
  });
  testVaddus4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 7},
  });
  testVavgs2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073816216},
      {{4294967295, 2147483647}, 1073741823},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgs4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073816088},
      {{4294967295, 2147483647}, 1073741823},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgu2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073848984},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgu4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1082237592},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVcmpeq2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpeq4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpges2Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 65535},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpges4Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpgeu2Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpgeu4Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpgts2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 65535},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgts4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgtu2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgtu4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmples2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 4294901760},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmples4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 4278190080},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmpleu2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmpleu4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmplts2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294901760},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmplts4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4278190080},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVcmpltu2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmpltu4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVcmpne2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmpne4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVhaddu2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 1073848984},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVhaddu4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 1065460376},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVmaxs2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147435825},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxs4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2130920753},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxu2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxu4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmins2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 262143},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVmins4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVminu2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 214321},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVminu4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 214321},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVneg2Cases({
      {214321, 4294818511},
      {3, 65533},
      {2147483647, 2147549185},
      {0, 0},
      {4294967295, 65537},
  });
  testVneg4Cases({
      {214321, 16628687},
      {3, 253},
      {2147483647, 2164326657},
      {0, 0},
      {4294967295, 16843009},
  });
  testVnegss2Cases({
      {214321, 4294818511},
      {3, 65533},
      {2147483647, 2147549185},
      {0, 0},
      {4294967295, 65537},
  });
  testVnegss4Cases({
      {214321, 16628687},
      {3, 253},
      {2147483647, 2164326657},
      {0, 0},
      {4294967295, 16843009},
  });
  testVsads2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 50478},
      // {{4294967295, 2147483647}, 32768}, // TODO: Need open after bug fixing.
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsads4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 251},
      // {{4294967295, 2147483647}, 128}, // TODO: Need open after bug fixing.
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsadu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 80586},
      {{4294967295, 2147483647}, 32768},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsadu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 771},
      {{4294967295, 2147483647}, 128},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVseteq2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVseteq4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetges2Cases({
      {{4, 3}, 65537},
      {{214321, 2147483647}, 1},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVsetges4Cases({
      {{4, 3}, 16843009},
      {{214321, 2147483647}, 65793},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetgeu2Cases({
      {{4, 3}, 65537},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65537},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVsetgeu4Cases({
      {{4, 3}, 16843009},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16843009},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetgts2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 1},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgts4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 65793},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgtu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgtu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetles2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 65536},
      {{4294967295, 2147483647}, 65537},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65537},
  });
  testVsetles4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 16777216},
      {{4294967295, 2147483647}, 16843009},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843009},
  });
  testVsetleu2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65537},
  });
  testVsetleu4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843009},
  });
  testVsetlts2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 65536},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetlts4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 16777216},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetltu2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetltu4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetne2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetne4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsub2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147763506},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVsub4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2164540978},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVsubss2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147763506},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVsubss4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2164540978},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVsubus2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsubus4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
