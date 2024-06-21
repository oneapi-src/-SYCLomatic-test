// ====------ thrust_unique_count.cu------------- *- CUDA -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>

void test_1() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  int count =
      thrust::unique_count(thrust::host, A, A + N, thrust::equal_to<int>());
  if (count != 4) {
    printf("test_1 run failed\n");
    exit(-1);
  }
  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  int count = thrust::unique_count(A, A + N, thrust::equal_to<int>());
  if (count != 4) {
    printf("test_2 run failed\n");
    exit(-1);
  }
  printf("test_2 run passed!\n");
}

void test_3() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  int count = thrust::unique_count(thrust::host, A, A + N);
  if (count != 4) {
    printf("test_3 run failed\n");
    exit(-1);
  }
  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  int count = thrust::unique_count(A, A + N);
  if (count != 4) {
    printf("test_4 run failed\n");
    exit(-1);
  }
  printf("test_4 run passed!\n");
}

void test_5() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};

  thrust::host_vector<int> h_A(A, A + N);

  int count = thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N,
                                   thrust::equal_to<int>());
  if (count != 4) {
    printf("test_5 run failed\n");
    exit(-1);
  }
  printf("test_5 run passed!\n");
}

void test_6() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};

  thrust::device_vector<int> d_A(A, A + N);

  int count = thrust::unique_count(thrust::device, d_A.begin(), d_A.begin() + N,
                                   thrust::equal_to<int>());
  if (count != 4) {
    printf("test_6 run failed\n");
    exit(-1);
  }
  printf("test_6 run passed!\n");
}

void test_7() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::host_vector<int> h_A(A, A + N);
  int count = thrust::unique_count(h_A.begin(), h_A.begin() + N,
                                   thrust::equal_to<int>());
  if (count != 4) {
    printf("test_7 run failed\n");
    exit(-1);
  }
  printf("test_7 run passed!\n");
}

void test_8() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::device_vector<int> d_A(A, A + N);
  int count = thrust::unique_count(d_A.begin(), d_A.begin() + N,
                                   thrust::equal_to<int>());
  if (count != 4) {
    printf("test_8 run failed\n");
    exit(-1);
  }
  printf("test_8 run passed!\n");
}

void test_9() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::host_vector<int> h_A(A, A + N);
  int count = thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N);
  if (count != 4) {
    printf("test_9 run failed\n");
    exit(-1);
  }
  printf("test_9 run passed!\n");
}

void test_10() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::device_vector<int> d_A(A, A + N);
  int count =
      thrust::unique_count(thrust::device, d_A.begin(), d_A.begin() + N);
  if (count != 4) {
    printf("test_10 run failed\n");
    exit(-1);
  }
  printf("test_10 run passed!\n");
}

void test_11() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::host_vector<int> h_A(A, A + N);
  int count = thrust::unique_count(h_A.begin(), h_A.begin() + N);
  if (count != 4) {
    printf("test_11 run failed\n");
    exit(-1);
  }
  printf("test_11 run passed!\n");
}

void test_12() {

  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::device_vector<int> d_A(A, A + N);
  int count = thrust::unique_count(d_A.begin(), d_A.begin() + N);
  if (count != 4) {
    printf("test_12 run failed\n");
    exit(-1);
  }
  printf("test_12 run passed!\n");
}

int main() {

  test_1();
  test_2();
  test_3();
  test_4();

  test_5();
  test_6();
  test_7();
  test_8();

  test_9();
  test_10();
  test_11();
  test_12();

  return 0;
}