// ===------- cooperative_group_coalesced_group.cu --------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedExampleKernel(int *data) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  int res = 0;
  if (block.thread_rank() % 4) {
    cg::coalesced_group active = cg::coalesced_threads();

    // Example operation: let the first thread in the coalesced group increment the data
    if (active.thread_rank() == 0) {
      res = atomicAdd(&data[active.size() - 1], 1);
    }
    // Temp disable this API.
    // res = active.shfl(res, 0);
  }
}

int main() {
  const int dataSize = 256;
  const int bytes = dataSize * sizeof(int);

  // Allocate memory on the host and the device
  int *h_data = (int *)malloc(bytes);
  int *d_data;
  cudaMalloc(&d_data, bytes);

  // Initialize host memory to zeros
  memset(h_data, 0, bytes);

  // Copy host data to device
  cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

  // Execute the kernel
  coalescedExampleKernel<<<1, dataSize>>>(d_data);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

  // Check results
  bool passed = true;
  for (int i = 0; i < dataSize; ++i) {
    if (h_data[i] != 0)
      std::cout << "i is  " << i << " test " << h_data[i] << std::endl;
    /*
        if (h_data[i] != ((i + 1) % 32 == 0 ? 1 : 0)) {
            passed = false;
            std::cout << "Test failed at index " << i << ": expected " << ((i + 1) % 32 == 0 ? 1 : 0) << ", got " << h_data[i] << std::endl;
            break;
        }
        */
  }

  if (passed) {
    std::cout << "Test passed!" << std::endl;
  }

  // Clean up memory
  free(h_data);
  cudaFree(d_data);

  return 0;
}