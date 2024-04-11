// ====------ thrust_advance_trans_op_itr.cu----- *- CUDA -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

struct square {
  __host__ __device__ int operator()(int x) const { return x * x; }
};

void test_1() {
  const int N = 5;
  thrust::device_vector<int> vec(N);
  thrust::sequence(vec.begin(), vec.end());

  auto output_iter =
      thrust::make_transform_output_iterator(vec.begin(), square());

  thrust::transform(vec.begin(), vec.end(), output_iter, square());

  int ans[N] = {0, 1, 16, 81, 256};

  for (int i = 0; i < N; i++) {
    if (vec[i] != ans[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  const int N = 5;
  thrust::device_vector<int> vec(N);
  thrust::sequence(vec.begin(), vec.end());

  thrust::device_vector<int>::iterator iter = vec.begin();
  thrust::advance(iter, 2); // Advance by 2 steps

  if (*iter != 2) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

int main() {
  test_1();
  test_2();

  return 0;
}