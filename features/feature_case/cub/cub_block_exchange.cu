// ===------- cub_block_exchange.cu---------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

__global__ void StripedToBlockedKernel(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockExchange(temp_storage).StripedToBlocked(thread_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void BlockedToStripedKernel(int *d_data) {

  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  BlockExchange(temp_storage).BlockedToStriped(thread_data, thread_data);
  cub::StoreDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

__global__ void ScatterToBlockedKernel(int *d_data, int *d_rank) {

  using BlockExchange = cub::BlockExchange<int, 128, 4>;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4], thread_rank[4];
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_rank, thread_rank);
  BlockExchange(temp_storage).ScatterToBlocked(thread_data, thread_rank);
  cub::StoreDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

__global__ void ScatterToStripedKernel(int *d_data, int *d_rank) {

  using BlockExchange = cub::BlockExchange<int, 128, 4>;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4], thread_rank[4];
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_rank, thread_rank);
  BlockExchange(temp_storage).ScatterToStriped(thread_data, thread_rank);
  cub::StoreDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

bool test_striped_to_blocked() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
  }

  StripedToBlockedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_striped_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << "test_striped_to_blocked pass\n";
  return true;
}

bool test_blocked_to_striped() {
  int *d_data, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  BlockedToStripedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

bool test_scatter_to_blocked() {
  int *d_data, *d_rank;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  cudaMallocManaged(&d_rank, sizeof(int) * 512);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
    d_rank[4 * i + 0] = i * 4 + 0;
    d_rank[4 * i + 1] = i * 4 + 1;
    d_rank[4 * i + 2] = i * 4 + 2;
    d_rank[4 * i + 3] = i * 4 + 3;
  }

  ScatterToBlockedKernel<<<1, 128>>>(d_data, d_rank);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_scatter_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << "test_scatter_to_blocked pass\n";
  return true;
}

bool test_scatter_to_striped() {
  int *d_data, *d_rank, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  cudaMallocManaged(&d_rank, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  d_rank[0] = 0;
  d_rank[128] = 1;
  d_rank[256] = 2;
  d_rank[384] = 3;
  for (int i = 1; i < 128; i++) {
    d_rank[0 * 128 + i] = d_rank[0 * 128 + i - 1] + 4;
    d_rank[1 * 128 + i] = d_rank[1 * 128 + i - 1] + 4;
    d_rank[2 * 128 + i] = d_rank[2 * 128 + i - 1] + 4;
    d_rank[3 * 128 + i] = d_rank[3 * 128 + i - 1] + 4;
  }

  ScatterToStripedKernel<<<1, 128>>>(d_data, d_rank);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i + 0 * 128;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

int main() {
  return !(test_blocked_to_striped() && test_striped_to_blocked() &&
           test_scatter_to_blocked() && test_scatter_to_striped());
}
