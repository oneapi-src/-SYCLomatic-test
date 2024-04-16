// ===------- asm_v2inst.cu -------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <string>
#include <time.h>
#include <tuple>
#include <vector>

#define PRINT_PASS 0
#define PRINT_CASE 0

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

void checkResult(const string &FuncName, const vector<long> &Inputs,
                 const int &Expect, const int &DeviceResult) {
  if (PRINT_CASE || (!PRINT_PASS && Expect == DeviceResult)) {
    passed++;
    return;
  }
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << DeviceResult << " (expect " << Expect << ")";
  check(Expect == DeviceResult);
}

__global__ void vadd2SUS(int *Result, int a, int b, int c) {
  asm("vadd2.s32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vadd2SUSSat(int *Result, int a, int b, int c) {
  asm("vadd2.s32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vadd2SUSAdd(int *Result, int a, int b, int c) {
  asm("vadd2.s32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVadd2SUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vadd2SUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vadd2.s32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vadd2SUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vadd2.s32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vadd2SUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vadd2.s32.u32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vsub2USU(int *Result, int a, int b, int c) {
  asm("vsub2.u32.s32.u32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vsub2USUSat(int *Result, int a, int b, int c) {
  asm("vsub2.u32.s32.u32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vsub2USUAdd(int *Result, int a, int b, int c) {
  asm("vsub2.u32.s32.u32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVsub2USU(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vsub2USU<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vsub2.u32.s32.u32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vsub2USUSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vsub2.u32.s32.u32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vsub2USUAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vsub2.u32.s32.u32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vabsdiff2SSU(int *Result, int a, int b, int c) {
  asm("vabsdiff2.s32.s32.u32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vabsdiff2SSUSat(int *Result, int a, int b, int c) {
  asm("vabsdiff2.s32.s32.u32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vabsdiff2SSUAdd(int *Result, int a, int b, int c) {
  asm("vabsdiff2.s32.s32.u32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVabsdiff2SSU(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vabsdiff2SSU<<<1, 1>>>(Result, get<0>(TestCase.first),
                           get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vabsdiff2.s32.s32.u32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vabsdiff2SSUSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                              get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vabsdiff2.s32.s32.u32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vabsdiff2SSUAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                              get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vabsdiff2.s32.s32.u32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vmin2USS(int *Result, int a, int b, int c) {
  asm("vmin2.u32.s32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmin2USSSat(int *Result, int a, int b, int c) {
  asm("vmin2.u32.s32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmin2USSAdd(int *Result, int a, int b, int c) {
  asm("vmin2.u32.s32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVmin2USS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vmin2USS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmin2.u32.s32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vmin2USSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmin2.u32.s32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vmin2USSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vmin2.u32.s32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vmax2UUS(int *Result, int a, int b, int c) {
  asm("vmax2.u32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmax2UUSSat(int *Result, int a, int b, int c) {
  asm("vmax2.u32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmax2UUSAdd(int *Result, int a, int b, int c) {
  asm("vmax2.u32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVmax2UUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vmax2UUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmax2.u32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vmax2UUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmax2.u32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vmax2UUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vmax2.u32.u32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vavrg2UUS(int *Result, int a, int b, int c) {
  asm("vavrg2.u32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vavrg2UUSSat(int *Result, int a, int b, int c) {
  asm("vavrg2.u32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vavrg2UUSAdd(int *Result, int a, int b, int c) {
  asm("vavrg2.u32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testvavrg2UUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vavrg2UUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vavrg2.u32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vavrg2UUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vavrg2.u32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vavrg2UUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vavrg2.u32.u32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

int main() {
  srand(unsigned(time(nullptr)));
  int a = rand();
  int b = rand();
  int c = rand();
  testVadd2SUS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {7, 7, 8}},
      {{30000000, 400000000, 10}, {429934464, 429934464, 24874}},
      {{INT16_MAX, 1, 100}, {32768, 32767, 32868}},
      {{UINT16_MAX, 1, 1000}, {0, 32767, 66536}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65534, 32767, 75534}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32766, 32766, 132766}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-2, 2147450879, 1131069}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-32770, -32770, 10032765}},
      {{INT16_MIN, INT32_MIN, 100000000}, {2147450880, 2147450879, 100065535}},
      {{454422046, 990649001, 541577428}, {1445005511, 1445036031, 541667260}},
      {{2123447767, 63088206, 406272673}, {-2108431323, 2147476517, 406298905}},
      {{1127203977, 209928516, 352777355}, {1337066957, 1337098239, 352864778}},
  });
  testVsub2USU({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {65535, 0, 0}},
      {{30000000, 400000000, 10}, {-370000000, 0, -54916}},
      {{INT16_MAX, 1, 100}, {32766, 32766, 32866}},
      {{UINT16_MAX, 1, 1000}, {65534, 0, 998}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {0, 0, -55536}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32768, 0, 67232}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-65536, 0, 934463}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {98304, 0, 9901697}},
      {{INT16_MIN, INT32_MIN, 100000000}, {2147450880, 0, 99934463}},
      {{454422046, 990649001, 541577428}, {-536226955, 0, 541557842}},
      {{2123447767, 63088206, 406272673}, {2060425097, 2060386304, 406277369}},
      {{1127203977, 209928516, 352777355}, {917275461, 917241856, 352759420}},
  });
  testVabsdiff2SSU({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {1, 1, 2}},
      {{30000000, 400000000, 10}, {370065536, 370049023, 54936}},
      {{INT16_MAX, 1, 100}, {32766, 32766, 32866}},
      {{UINT16_MAX, 1, 1000}, {2, 2, 1002}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {0, 32767, 75536}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32768, 32767, 132768}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {65536, 98303, 1065537}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-32768, 2147450879, 10098303}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-2147385344, 2147450879, 100065537}},
      {{454422046, 990649001, 541577428}, {536292491, 536292491, 541597014}},
      {{2123447767, 63088206, 406272673}, {2060413047, 2060413047, 406330855}},
      {{1127203977, 209928516, 352777355}, {917273787, 917273787, 352823282}},
  });
  testVmin2USS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {3, 3, 4}},
      {{30000000, 400000000, 10}, {29983744, 29949952, -31277}},
      {{INT16_MAX, 1, 100}, {1, 1, 101}},
      {{UINT16_MAX, 1, 1000}, {65535, 0, 999}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65535, 0, 9999}},
      {{INT16_MAX, UINT16_MAX, 100000}, {65535, 0, 99999}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-1, 0, 999998}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-1, 0, 9999998}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-2147450880, 0, 99934464}},
      {{454422046, 990649001, 541577428}, {454422046, 454361088, 541579783}},
      {{2123447767, 63088206, 406272673}, {63088206, 63045632, 406250673}},
      {{1127203977, 209928516, 352777355}, {209962121, 209911808, 352765335}},
  });
  testVmax2UUS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {4, 4, 5}},
      {{30000000, 400000000, 10}, {400016256, 400016256, 56161}},
      {{INT16_MAX, 1, 100}, {32767, 32767, 32867}},
      {{UINT16_MAX, 1, 1000}, {65535, 65535, 66535}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65535, 65535, 75535}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32767, 32767, 132767}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-1, -1, 1131070}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {32767, 32767, 10032767}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-32768, -32768, 100098303}},
      {{454422046, 990649001, 541577428}, {990703134, 990703134, 541653502}},
      {{2123447767, 63088206, 406272673}, {2123447767, 2123447767, 406320905}},
      {{1127203977, 209928516, 352777355}, {1127203977, 1127203977, 352844867}},
  });
  testvavrg2UUS({
    // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {4, 4, 5}},
      {{30000000, 400000000, 10}, {214967232, 214967232, 12442}},
      {{INT16_MAX, 1, 100}, {16384, 16384, 16484}},
      {{UINT16_MAX, 1, 1000}, {32768, 32768, 33768}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {32767, 32767, 42767}},
      {{INT16_MAX, UINT16_MAX, 100000}, {16383, 16383, 116383}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-2147450881, -2147450881, 1065535}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-49153, 16383, 10016382}},
      {{INT16_MIN, INT32_MIN, 100000000}, {1073758208, 1073758208, 100032768}},
      {{454422046, 990649001, 541577428}, {722568292, 722568292, 541622345}},
      {{2123447767, 63088206, 406272673}, {1093333522, 1093271552, 406285789}},
      {{1127203977, 209928516, 352777355}, {668566247, 668566247, 352821067}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
