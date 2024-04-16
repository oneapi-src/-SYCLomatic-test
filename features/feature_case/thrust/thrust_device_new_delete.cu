// ====------ thrust_device_new_delete.cu--------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/uninitialized_fill.h>

void test_1() {
  const int N = 137;
  int val = 46;
  thrust::device_ptr<int> d_array = thrust::device_new<int>(N);
  thrust::uninitialized_fill(d_array, d_array + N, val);

  thrust::host_vector<int> hostVec(d_array, d_array + N);

  thrust::device_delete(d_array, N);

  for (int i = 0; i < N; i++) {
    if (hostVec[i] != 46) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 137;
  int val = 46;

  thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(N);

  thrust::device_ptr<int> a;
  thrust::device_ptr<int> b;
  b = a;

  thrust::device_ptr<int> d_array = thrust::device_new<int>(d_mem, N);
  thrust::uninitialized_fill(thrust::device, d_array, d_array + N, val);

  thrust::host_vector<int> hostVec(d_array, d_array + N);

  for (int i = 0; i < N; i++) {
    if (hostVec[i] != 46) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  thrust::device_delete(d_array, N);
  printf("test_2 run passed!\n");
}

void test_3() {

  const int N = 137;

  int val = 46;
  thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(N);
  thrust::device_ptr<int> d_array = thrust::device_new<int>(d_mem, val, N);

  thrust::host_vector<int> hostVec(d_array, d_array + N);

  for (int i = 0; i < N; i++) {
    if (hostVec[i] != 46) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  thrust::device_delete(d_array, N);
  printf("test_3 run passed!\n");
}

int main() {

  test_1();
  test_2();
  test_3();

  return 0;
}
