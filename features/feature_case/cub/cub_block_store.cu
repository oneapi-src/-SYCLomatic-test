// ====------ cub_block_store.cu---------- *- CUDA -* ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>

bool verify_data(int *data, int *expect, int num, int step = 1) {
  for (int i = 0; i < num; i = i + step) {
    if (data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

void print_data(int *data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if ((i + 1) % 32 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

__global__ void BlockedKernel(int *d_data, int valid_items) {
  // Specialize BlockStore for a 1D block of 32 threads owning 4 integer items
  // each
  using BlockStore = cub::BlockStore<int, 32, 4>;

  __shared__ typename BlockStore::TempStorage temp_storage;

  int thread_data[4];
  thread_data[0] = threadIdx.x * 4 + 0;
  thread_data[1] = threadIdx.x * 4 + 1;
  thread_data[2] = threadIdx.x * 4 + 2;
  thread_data[3] = threadIdx.x * 4 + 3;

  BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
}

__global__ void StripedKernel(int *d_data, int valid_items) {
  // Specialize BlockStore for a 1D block of 32 threads owning 4 integer items
  // each
  using BlockStore = cub::BlockStore<int, 32, 4, cub::BLOCK_STORE_STRIPED>;

  __shared__ typename BlockStore::TempStorage temp_storage;

  int thread_data[4];
  thread_data[0] = threadIdx.x * 4 + 0;
  thread_data[1] = threadIdx.x * 4 + 1;
  thread_data[2] = threadIdx.x * 4 + 2;
  thread_data[3] = threadIdx.x * 4 + 3;
  BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
}

int main() {
  int *d_data;
  bool result = true;
  int ref_data_1[32] = {0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int ref_data_2[32] = {0, 4, 8, 12, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int h_data[32] = {0};
  cudaMalloc(&d_data, sizeof(int) * 32);
  cudaMemset(d_data, 0, sizeof(int) * 32);
  BlockedKernel<<<1, 32>>>(d_data, 5);
  cudaStreamSynchronize(0);
  cudaMemcpy(h_data, d_data, sizeof(int)*32, cudaMemcpyDeviceToHost);

  if (!verify_data(h_data, ref_data_1, 32)) {
    std::cout << "BlockExclusiveScanKernel"
              << " verify failed" << std::endl;
    result = false;
    std::cout << "expect:" << std::endl;
    print_data(ref_data_1, 32);
    std::cout << "current result:" << std::endl;
    print_data(d_data, 32);
  }

  cudaMemset(d_data, 0, sizeof(int) * 31);
  memset(h_data, 0, sizeof(int) * 32);
  StripedKernel<<<1, 32>>>(d_data, 5);
  cudaStreamSynchronize(0);
  cudaMemcpy(h_data, d_data, sizeof(int)*32, cudaMemcpyDeviceToHost);

  if (!verify_data(h_data, ref_data_2, 32)) {
    std::cout << "BlockExclusiveScanKernel"
              << " verify failed" << std::endl;
    result = false;
    std::cout << "expect:" << std::endl;
    print_data(ref_data_1, 32);
    std::cout << "current result:" << std::endl;
    print_data(d_data, 32);
  }

  if (result) {
    std::cout << "cub_block_store passed!" << std::endl;
    return 0;
  }
  return 1;
}
