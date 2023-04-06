// ====------ thrust_reverse.cu-------------------------------------- *- CUDA -*
// -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reverse.h>

void test_1() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::reverse(thrust::device, device_data.begin(), device_data.end());

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_1 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_1 run passed!\n";
}

void test_2() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::host_vector<int> host_data(data, data + N);
  thrust::reverse(thrust::host, host_data.begin(), host_data.end());

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_2 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_2 run passed!\n";
}

void test_3() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::reverse(thrust::host, data, data + N);

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_3 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_3 run passed!\n";
}

void test_4() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::reverse(device_data.begin(), device_data.end());

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_4 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_4 run passed!\n";
}

void test_5() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::host_vector<int> host_data(data, data + N);
  thrust::reverse(host_data.begin(), host_data.end());

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_5 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_5 run passed!\n";
}

void test_6() {

  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::reverse(data, data + N);

  int ref[] = {5, 4, 3, 2, 1, 0};

  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_6 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_6 run passed!\n";
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();

  return 0;
}
