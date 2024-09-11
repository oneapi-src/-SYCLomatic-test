// ====------ thrust-sort_by_key.cu ---------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

void test_1() {
  int keys[7]    =  {7,  5,  4,  6,  2,  1,  3};
  char values[7] = {'B','G','F','A','D','C','E'};
  thrust::device_vector<int> d_keys(keys, keys + 7);
  thrust::device_vector<char> d_values(values, values + 7);
  thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
  char values_ref[7] = {'C','D','E','F','G','A','B'};
  for (int i = 0; i < d_values.size(); i++) {
    if (d_values[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_1 run passed!\n");
}

void test_2() {
  int keys[7]    =  {7,  5,  4,  6,  2,  1,  3};
  char values[7] = {'B','G','F','A','D','C','E'};
  thrust::device_vector<int> d_keys(keys, keys + 7);
  thrust::device_vector<char> d_values(values, values + 7);
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
  char values_ref[7] = {'C','D','E','F','G','A','B'};
  for (int i = 0; i < d_values.size(); i++) {
    if (d_values[i] != values_ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int keys[7]    =  {7,  5,  4,  6,  2,  1,  3};
  char values[7] = {'B','G','F','A','D','C','E'};
  thrust::device_vector<int> d_keys(keys, keys + 7);
  thrust::device_vector<char> d_values(values, values + 7);
  thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<int>());
  char values_ref[7] = {'B','A','G','F','E','D','C'};
  for (int i = 0; i < d_values.size(); i++) {
    if (d_values[i] != values_ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int keys[7]    =  {7,  5,  4,  6,  2,  1,  3};
  char values[7] = {'B','G','F','A','D','C','E'};
  thrust::device_vector<int> d_keys(keys, keys + 7);
  thrust::device_vector<char> d_values(values, values + 7);
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<int>());
  char values_ref[7] = {'C','D','E','F','G','A','B'};
  for (int i = 0; i < d_values.size(); i++) {
    if (d_values[i] != values_ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run passed!\n");
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  return 0;
}
