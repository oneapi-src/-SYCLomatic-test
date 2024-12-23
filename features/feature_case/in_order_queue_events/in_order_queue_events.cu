// ====------ in_order_queue_events.cu--------------- *- CUDA -*---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const int *A, int *B, int *C, int N) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N) {
          C[i] = A[i] + B[i];
      }
}

void test1() {
  cudaStream_t s1;
  cudaStreamCreate(&s1);

  int N = 10;
  size_t size = N * sizeof(int);

  int *h_A = new int[N];
  int *h_B = new int[N];
  int *h_C = new int[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<int>(i);
    h_B[i] = static_cast<int>(i * 2);
  }

  int *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, s1);

  vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }
}

void test2() {
  cudaStream_t s1;
  cudaStreamCreate(&s1);

  int N = 10;
  size_t size = N * sizeof(int);

  int *h_A = new int[N];
  int *h_B = new int[N];
  int *h_C = new int[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<int>(i);
    h_B[i] = static_cast<int>(i * 2);
  }

  int *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, 0);

  vectorAdd<<<1, N, 0, s1>>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }
}

int main() {
  test1();
  test2();
  std::cout << "test pass" << std::endl;
  return 0;
}