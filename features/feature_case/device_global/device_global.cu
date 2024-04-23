// ====------ device_global.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__constant__ int var = 10;
__constant__ int arr[2] = {1, 2};

__global__ void kernel(int *ptr) {
  *ptr = var + arr[1];
}

int main() {
    int *dev;
    cudaMallocManaged(&dev, sizeof(int));
    *dev = 0;
    kernel<<<1, 1>>>(dev);
    cudaDeviceSynchronize();
    if(*dev != 12) {
      std::cout << "fail" << std::endl;
      exit(-1);
    }
    std::cout << "pass" << std::endl;
    return 0;
}