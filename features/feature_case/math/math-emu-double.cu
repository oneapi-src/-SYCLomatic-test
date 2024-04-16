// ====------------ math-emu-double.cu---------- *- CUDA -* -------------===////
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

using namespace std;

typedef vector<double> d_vector;
typedef tuple<double, double, double> d_tuple3;
typedef tuple<double, double, double, double> d_tuple4;
typedef pair<double, int> di_pair;

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

template <typename T = double>
void checkResult(const string &FuncName, const vector<T> &Inputs,
                 const double &Expect, const double &DeviceResult,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision) << DeviceResult
       << " (expect " << Expect - pow(10, -precision) << " ~ "
       << Expect + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(DeviceResult - Expect) < pow(10, -precision));
}

__global__ void setVecValue(double *Input1, const double Input2) {
  *Input1 = Input2;
}

// Double Precision Mathematical Functions

__global__ void _norm(double *const DeviceResult, int Input1,
                      const double *Input2) {
  *DeviceResult = norm(Input1, Input2);
}

void testNorm(double *const DeviceResult, int Input1, const double *Input2) {
  _norm<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // TODO: Need test host side.
}

void testNormCases(const vector<pair<d_vector, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testNorm(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("norm", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _norm3d(double *const DeviceResult, double Input1,
                        double Input2, double Input3) {
  *DeviceResult = norm3d(Input1, Input2, Input3);
}

void testNorm3d(double *const DeviceResult, double Input1, double Input2,
                double Input3) {
  _norm3d<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNorm3dCases(const vector<pair<d_tuple3, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNorm3d(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
               get<2>(TestCase.first));
    checkResult("norm3d",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

__global__ void _norm4d(double *const DeviceResult, double Input1,
                        double Input2, double Input3, double Input4) {
  *DeviceResult = norm4d(Input1, Input2, Input3, Input4);
}

void testNorm4d(double *const DeviceResult, double Input1, double Input2,
                double Input3, double Input4) {
  _norm4d<<<1, 1>>>(DeviceResult, Input1, Input2, Input3, Input4);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNorm4dCases(const vector<pair<d_tuple4, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNorm4d(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
               get<2>(TestCase.first), get<3>(TestCase.first));
    checkResult("norm4d",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first), get<3>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

__global__ void _normcdf(double *const DeviceResult, double Input) {
  *DeviceResult = normcdf(Input);
}

void testNormcdf(double *const DeviceResult, double Input) {
  _normcdf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormcdfCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdf(DeviceResult, TestCase.first);
    checkResult("normcdf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _rcbrt(double *const DeviceResult, double Input1) {
  *DeviceResult = rcbrt(Input1);
}

void testRcbrtCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    _rcbrt<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("rcbrt", {TestCase.first}, TestCase.second.first, *DeviceResult,
                TestCase.second.second);
  }
}

__global__ void _rnorm(double *const DeviceResult, int Input1,
                       const double *Input2) {
  *DeviceResult = rnorm(Input1, Input2);
}

void testRnorm(double *const DeviceResult, int Input1, const double *Input2) {
  _rnorm<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnormCases(const vector<pair<d_vector, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testRnorm(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("rnorm", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _rnorm3d(double *const DeviceResult, double Input1,
                         double Input2, double Input3) {
  *DeviceResult = rnorm3d(Input1, Input2, Input3);
}

void testRnorm3d(double *const DeviceResult, double Input1, double Input2,
                 double Input3) {
  _rnorm3d<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnorm3dCases(const vector<pair<d_tuple3, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testRnorm3d(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                get<2>(TestCase.first));
    checkResult("rnorm3d",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

__global__ void _rnorm4d(double *const DeviceResult, double Input1,
                         double Input2, double Input3, double Input4) {
  *DeviceResult = rnorm4d(Input1, Input2, Input3, Input4);
}

void testRnorm4d(double *const DeviceResult, double Input1, double Input2,
                 double Input3, double Input4) {
  _rnorm4d<<<1, 1>>>(DeviceResult, Input1, Input2, Input3, Input4);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnorm4dCases(const vector<pair<d_tuple4, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testRnorm4d(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                get<2>(TestCase.first), get<3>(TestCase.first));
    checkResult("rnorm4d",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first), get<3>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

// Double Precision Intrinsics

__global__ void dadd_rd(double *const Result, double Input1, double Input2) {
  *Result = __dadd_rd(Input1, Input2);
}

void testDadd_rdCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dadd_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dadd_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dadd_rn(double *const Result, double Input1, double Input2) {
  *Result = __dadd_rn(Input1, Input2);
}

void testDadd_rnCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dadd_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dadd_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dadd_ru(double *const Result, double Input1, double Input2) {
  *Result = __dadd_ru(Input1, Input2);
}

void testDadd_ruCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dadd_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dadd_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dadd_rz(double *const Result, double Input1, double Input2) {
  *Result = __dadd_rz(Input1, Input2);
}

void testDadd_rzCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dadd_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dadd_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dmul_rd(double *const Result, double Input1, double Input2) {
  *Result = __dmul_rd(Input1, Input2);
}

void testDmul_rdCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dmul_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dmul_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dmul_rn(double *const Result, double Input1, double Input2) {
  *Result = __dmul_rn(Input1, Input2);
}

void testDmul_rnCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dmul_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dmul_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dmul_ru(double *const Result, double Input1, double Input2) {
  *Result = __dmul_ru(Input1, Input2);
}

void testDmul_ruCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dmul_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dmul_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dmul_rz(double *const Result, double Input1, double Input2) {
  *Result = __dmul_rz(Input1, Input2);
}

void testDmul_rzCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dmul_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dmul_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dsub_rd(double *const Result, double Input1, double Input2) {
  *Result = __dsub_rd(Input1, Input2);
}

void testDsub_rdCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dsub_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dsub_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dsub_rn(double *const Result, double Input1, double Input2) {
  *Result = __dsub_rn(Input1, Input2);
}

void testDsub_rnCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dsub_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dsub_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dsub_ru(double *const Result, double Input1, double Input2) {
  *Result = __dsub_ru(Input1, Input2);
}

void testDsub_ruCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dsub_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dsub_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void dsub_rz(double *const Result, double Input1, double Input2) {
  *Result = __dsub_rz(Input1, Input2);
}

void testDsub_rzCases(
    const vector<pair<pair<double, double>, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    dsub_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__dsub_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testNormCases({
      {{-0.3, -0.34, -0.98}, {1.079814798935447, 15}},
      {{0.3, 0.34, 0.98}, {1.079814798935447, 15}},
      {{0.5}, {0.5, 16}},
      {{23, 432, 23, 456, 23}, {629.4020972319682, 13}},
  });
  testNorm3dCases({
      {{-0.3, -0.34, -0.98}, {1.079814798935447, 15}},
      {{0.3, 0.34, 0.98}, {1.079814798935447, 15}},
      {{0.5, 456, 23}, {456.5799491874342, 13}},
      {{23, 432, 23}, {433.222806417206, 13}},
  });
  testNorm4dCases({
      {{-0.3, -0.34, -0.98, 1}, {1.471733671558818, 15}},
      {{0.3, 0.34, 0.98, 1}, {1.471733671558818, 15}},
      {{0.5, 456, 23, 1}, {456.5810442845827, 13}},
      {{23, 432, 23, 1}, {433.2239605562001, 13}},
  });
  testNormcdfCases({
      {-5, {0.000000286651571879194, 21}},
      {-3, {0.001349898031630095, 18}},
      {0, {0.5, 16}},
      {1, {0.841344746068543, 15}},
      {5, {0.9999997133484281, 16}},
  });
  testRcbrtCases({
      {-0.3, {-1.493801582185722, 15}},
      {0.3, {1.493801582185722, 15}},
      {0.5, {1.259921049894873, 15}},
      {23, {0.3516338869169593, 16}},
  });
  testRnormCases({
      {{-0.3, -0.34, -0.98}, {0.926084733220795, 15}},
      {{0.3, 0.34, 0.98}, {0.926084733220795, 15}},
      {{0.5}, {2, 15}},
      {{23, 432, 23, 456, 23}, {0.001588809450108087, 18}},
  });
  testRnorm3dCases({
      {{-0.3, -0.34, -0.98}, {0.926084733220795, 15}},
      {{0.3, 0.34, 0.98}, {0.926084733220795, 15}},
      {{0.5, 456, 23}, {0.002190196923407782, 18}},
      {{23, 432, 23}, {0.002308281062740199, 18}},
  });
  testRnorm4dCases({
      {{-0.3, -0.34, -0.98, 1}, {0.679470762492529, 15}},
      {{0.3, 0.34, 0.98, 1}, {0.679470762492529, 15}},
      {{0.5, 456, 23, 1}, {0.002190191670280358, 18}},
      {{23, 432, 23, 1}, {0.002308274913317669, 18}},
  });
  testDadd_rdCases({
      {{-0.3, -0.4}, {-0.7, 7}},
      {{0.3, -0.4}, {-0.1, 8}},
      {{0.3, 0.4}, {0.7, 7}},
      {{0.3, 0.8}, {1.1, 7}},
      {{3, 4}, {7, 37}},
  });
  testDadd_rnCases({
      {{-0.3, -0.4}, {-0.7, 7}},
      {{0.3, -0.4}, {-0.1, 8}},
      {{0.3, 0.4}, {0.7, 7}},
      {{0.3, 0.8}, {1.1, 7}},
      {{3, 4}, {7, 37}},
  });
  testDadd_ruCases({
      {{-0.3, -0.4}, {-0.7, 7}},
      {{0.3, -0.4}, {-0.1, 8}},
      {{0.3, 0.4}, {0.7, 7}},
      {{0.3, 0.8}, {1.1, 7}},
      {{3, 4}, {7, 37}},
  });
  testDadd_rzCases({
      {{-0.3, -0.4}, {-0.7, 7}},
      {{0.3, -0.4}, {-0.1, 8}},
      {{0.3, 0.4}, {0.7, 7}},
      {{0.3, 0.8}, {1.1, 7}},
      {{3, 4}, {7, 37}},
  });
  testDmul_rdCases({
      {{-0.3, -0.4}, {0.12, 8}},
      {{0.3, -0.4}, {-0.12, 8}},
      {{0.3, 0.4}, {0.12, 8}},
      {{0.3, 0.8}, {0.24, 8}},
      {{3, 4}, {12, 37}},
  });
  testDmul_rnCases({
      {{-0.3, -0.4}, {0.12, 8}},
      {{0.3, -0.4}, {-0.12, 8}},
      {{0.3, 0.4}, {0.12, 8}},
      {{0.3, 0.8}, {0.24, 8}},
      {{3, 4}, {12, 37}},
  });
  testDmul_ruCases({
      {{-0.3, -0.4}, {0.12, 8}},
      {{0.3, -0.4}, {-0.12, 8}},
      {{0.3, 0.4}, {0.12, 8}},
      {{0.3, 0.8}, {0.24, 8}},
      {{3, 4}, {12, 37}},
  });
  testDmul_rzCases({
      {{-0.3, -0.4}, {0.12, 8}},
      {{0.3, -0.4}, {-0.12, 8}},
      {{0.3, 0.4}, {0.12, 8}},
      {{0.3, 0.8}, {0.24, 8}},
      {{3, 4}, {12, 37}},
  });
  testDsub_rdCases({
      {{-0.3, -0.4}, {0.1, 8}},
      {{0.3, -0.4}, {0.7, 7}},
      {{0.3, 0.4}, {-0.1, 8}},
      {{0.3, 0.8}, {-0.5, 15}},
      {{3, 4}, {-1, 37}},
  });
  testDsub_rnCases({
      {{-0.3, -0.4}, {0.1, 8}},
      {{0.3, -0.4}, {0.7, 7}},
      {{0.3, 0.4}, {-0.1, 8}},
      {{0.3, 0.8}, {-0.5, 37}},
      {{3, 4}, {-1, 37}},
  });
  testDsub_ruCases({
      {{-0.3, -0.4}, {0.1, 8}},
      {{0.3, -0.4}, {0.7, 7}},
      {{0.3, 0.4}, {-0.1, 8}},
      {{0.3, 0.8}, {-0.5, 37}},
      {{3, 4}, {-1, 37}},
  });
  testDsub_rzCases({
      {{-0.3, -0.4}, {0.1, 8}},
      {{0.3, -0.4}, {0.7, 7}},
      {{0.3, 0.4}, {-0.1, 8}},
      {{0.3, 0.8}, {-0.5, 37}},
      {{3, 4}, {-1, 37}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
