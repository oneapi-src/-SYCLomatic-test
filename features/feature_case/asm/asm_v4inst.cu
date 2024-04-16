// ===------- asm_v4inst.cu -------------------------------- *- CUDA -* ---===//
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

__global__ void vadd4SUS(int *Result, int a, int b, int c) {
  asm("vadd4.s32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vadd4SUSSat(int *Result, int a, int b, int c) {
  asm("vadd4.s32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vadd4SUSAdd(int *Result, int a, int b, int c) {
  asm("vadd4.s32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVadd4SUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vadd4SUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vadd4.s32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vadd4SUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vadd4.s32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vadd4SUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vadd4.s32.u32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vsub4USU(int *Result, int a, int b, int c) {
  asm("vsub4.u32.s32.u32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vsub4USUSat(int *Result, int a, int b, int c) {
  asm("vsub4.u32.s32.u32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vsub4USUAdd(int *Result, int a, int b, int c) {
  asm("vsub4.u32.s32.u32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVsub4USU(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vsub4USU<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vsub4.u32.s32.u32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vsub4USUSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vsub4.u32.s32.u32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vsub4USUAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vsub4.u32.s32.u32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vabsdiff4SSU(int *Result, int a, int b, int c) {
  asm("vabsdiff4.s32.s32.u32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vabsdiff4SSUSat(int *Result, int a, int b, int c) {
  asm("vabsdiff4.s32.s32.u32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vabsdiff4SSUAdd(int *Result, int a, int b, int c) {
  asm("vabsdiff4.s32.s32.u32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVabsdiff4SSU(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vabsdiff4SSU<<<1, 1>>>(Result, get<0>(TestCase.first),
                           get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vabsdiff4.s32.s32.u32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vabsdiff4SSUSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                              get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vabsdiff4.s32.s32.u32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vabsdiff4SSUAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                              get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vabsdiff4.s32.s32.u32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vmin4USS(int *Result, int a, int b, int c) {
  asm("vmin4.u32.s32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmin4USSSat(int *Result, int a, int b, int c) {
  asm("vmin4.u32.s32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmin4USSAdd(int *Result, int a, int b, int c) {
  asm("vmin4.u32.s32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVmin4USS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vmin4USS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmin4.u32.s32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vmin4USSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmin4.u32.s32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vmin4USSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vmin4.u32.s32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vmax4UUS(int *Result, int a, int b, int c) {
  asm("vmax4.u32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmax4UUSSat(int *Result, int a, int b, int c) {
  asm("vmax4.u32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vmax4UUSAdd(int *Result, int a, int b, int c) {
  asm("vmax4.u32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVmax4UUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vmax4UUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmax4.u32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vmax4UUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vmax4.u32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vmax4UUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vmax4.u32.u32.s32.add",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<2>(TestCase.second), *Result);
  }
}

__global__ void vavrg4UUS(int *Result, int a, int b, int c) {
  asm("vavrg4.u32.u32.s32 %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vavrg4UUSSat(int *Result, int a, int b, int c) {
  asm("vavrg4.u32.u32.s32.sat %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

__global__ void vavrg4UUSAdd(int *Result, int a, int b, int c) {
  asm("vavrg4.u32.u32.s32.add %0, %1, %2, %3;"
      : "=r"(*Result)
      : "r"(a), "r"(b), "r"(c));
}

void testVavrg4UUS(
    const vector<pair<tuple<int, int, int>, tuple<int, int, int>>> &TestCases) {
  int *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    string newCase = "{{" + to_string(get<0>(TestCase.first)) + ", " +
                     to_string(get<1>(TestCase.first)) + ", " +
                     to_string(get<2>(TestCase.first)) + "}, {";
    vavrg4UUS<<<1, 1>>>(Result, get<0>(TestCase.first), get<1>(TestCase.first),
                       get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vavrg4.u32.u32.s32",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<0>(TestCase.second), *Result);
    vavrg4UUSSat<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + ", ";
    checkResult("vavrg4.u32.u32.s32.sat",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                get<1>(TestCase.second), *Result);
    vavrg4UUSAdd<<<1, 1>>>(Result, get<0>(TestCase.first),
                          get<1>(TestCase.first), get<2>(TestCase.first));
    cudaDeviceSynchronize();
    newCase += to_string(*Result) + "}},";
    if (PRINT_CASE)
      cout << newCase << endl;
    checkResult("vavrg4.u32.u32.s32.add",
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
  testVadd4SUS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {7, 7, 8}},
      {{30000000, 400000000, 10}, {413157248, 410994559, 393}},
      {{INT16_MAX, 1, 100}, {32512, 32639, 483}},
      {{UINT16_MAX, 1, 1000}, {65280, 32639, 1511}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65278, 32639, 10508}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32510, 32383, 100380}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-258, 2139062143, 1001018}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-33026, -33153, 10000378}},
      {{INT16_MIN, INT32_MIN, 100000000}, {2147450880, 2139062016, 100000510}},
      {{454422046, 990649001, 541577428}, {1445005511, 1445035975, 541577754}},
      {{2123447767, 63088206, 406272673}, {-2125208795, 2136204159, 406273149}},
      {{1127203977, 209928516, 352777355}, {1337066957, 1337098111, 352777822}},
  });
  testVsub4USU({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {255, 0, 0}},
      {{30000000, 400000000, 10}, {-353222784, 0, -603}},
      {{INT16_MAX, 1, 100}, {32766, 32512, 225}},
      {{UINT16_MAX, 1, 1000}, {65534, 0, 997}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {0, 0, 9488}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32768, 0, 99616}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-65536, 0, 999486}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {16875520, 0, 9999106}},
      {{INT16_MIN, INT32_MIN, 100000000}, {2147450880, 0, 99999742}},
      {{454422046, 990649001, 541577428}, {-536226699, 589824, 541577222}},
      {{2123447767, 63088206, 406272673}, {2077202313, 2063597568, 406272267}},
      {{1127203977, 209928516, 352777355}, {934052677, 922746880, 352777014}},
  });
  testVabsdiff4SSU({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {1, 1, 2}},
      {{30000000, 400000000, 10}, {370065792, 377454463, 623}},
      {{INT16_MAX, 1, 100}, {32514, 32514, 229}},
      {{UINT16_MAX, 1, 1000}, {258, 258, 1003}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {0, 32639, 10512}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32768, 32639, 100384}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {16842752, 16875391, 1000514}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-32768, 2139062143, 10000894}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-2130608128, 2130804480, 100000258}},
      {{454422046, 990649001, 541577428}, {537472139, 537472127, 541577652}},
      {{2123447767, 63088206, 406272673}, {2066835831, 2071947639, 406273325}},
      {{1127203977, 209928516, 352777355}, {928284091, 928284031, 352777806}},
  });
  testVmin4USS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {3, 3, 4}},
      {{30000000, 400000000, 10}, {29983872, 16777216, -296}},
      {{INT16_MAX, 1, 100}, {255, 0, 99}},
      {{UINT16_MAX, 1, 1000}, {65535, 0, 998}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65535, 0, 9998}},
      {{INT16_MAX, UINT16_MAX, 100000}, {65535, 0, 99998}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-1, 0, 999996}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-1, 0, 9999996}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-2130739200, 0, 99999743}},
      {{454422046, 990649001, 541577428}, {453832361, 453771264, 541577362}},
      {{2123447767, 63088206, 406272673}, {59877079, 50331648, 406272434}},
      {{1127203977, 209928516, 352777355}, {209962121, 201326592, 352777063}},
  });
  testVmax4UUS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {4, 4, 5}},
      {{30000000, 400000000, 10}, {399098752, 399098752, 557}},
      {{INT16_MAX, 1, 100}, {32767, 32767, 482}},
      {{UINT16_MAX, 1, 1000}, {65535, 65535, 1510}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {65535, 65535, 10510}},
      {{INT16_MAX, UINT16_MAX, 100000}, {32767, 32767, 100382}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-1, -1, 1001020}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {32767, 32767, 10000382}},
      {{INT16_MIN, INT32_MIN, 100000000}, {-32768, -32768, 100000638}},
      {{454422046, 990649001, 541577428}, {991292958, 991292958, 541577776}},
      {{2123447767, 63088206, 406272673}, {2123447767, 2123447767, 406273220}},
      {{1127203977, 209928516, 352777355}, {1127203977, 1127203977, 352777802}},
  });
  testVavrg4UUS({
      // {{a, b, c}, {0, 0, 0}},
      {{3, 4, 1}, {4, 4, 5}},
      {{30000000, 400000000, 10}, {206578752, 206578752, 202}},
      {{INT16_MAX, 1, 100}, {16512, 16512, 292}},
      {{UINT16_MAX, 1, 1000}, {32896, 32896, 1256}},
      {{UINT16_MAX, UINT16_MAX, 10000}, {32639, 32639, 10254}},
      {{INT16_MAX, UINT16_MAX, 100000}, {16255, 16255, 100190}},
      {{UINT32_MAX, UINT16_MAX, 1000000}, {-2139062401, -2139062401, 1000510}},
      {{INT16_MAX, UINT32_MAX, 10000000}, {-49281, 16255, 10000188}},
      {{INT16_MIN, INT32_MIN, 100000000}, {1082146816, 1082146816, 100000256}},
      {{454422046, 990649001, 541577428}, {722568419, 722568192, 541577591}},
      {{2123447767, 63088206, 406272673}, {1093333395, 1093271699, 406272912}},
      {{1127203977, 209928516, 352777355}, {685343591, 671122279, 352777590}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
