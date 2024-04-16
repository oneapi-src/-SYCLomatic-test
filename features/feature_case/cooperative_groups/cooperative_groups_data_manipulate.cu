// ====------ cooperative_groups_data_manipulate.cu --------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cooperative_groups.h>
#include <cooperative_groups/details/scan.h>
#include <cstdio>
#include <cuda.h>
#include <iostream>
namespace cg = cooperative_groups;

__global__ void test_thread_num_kernel(int *out) {

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
  cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
  cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);
  cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
  cg::thread_block_tile<2> tile2 = cg::tiled_partition<2>(block);
  if (threadIdx.x == 1) {
    out[0] = tile32.num_threads();
    out[1] = tile16.num_threads();
    out[2] = tile8.num_threads();
    out[3] = tile4.num_threads();
    out[4] = tile2.num_threads();
    out[5] = block.num_threads();
  }
}
bool test_thread_num() {
  int num_elements = 6;
  int expected[] = {32, 16, 8, 4, 2, 56};
  int *output;
  int *result;

  result = (int *)malloc(num_elements * sizeof(int));
  cudaMalloc((void **)&output, num_elements * sizeof(int));
  test_thread_num_kernel<<<1, 56>>>(output);
  cudaMemcpy(result, output, sizeof(unsigned int) * num_elements,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < num_elements; i++) {
    if (expected[i] != result[i]) {
      return false;
    }
  }
  return true;
}

__global__ void testExclusive(unsigned int *output) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(thread_block);
  unsigned int val[7];
  val[0] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::plus<unsigned int>());

  val[1] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::less<unsigned int>());

  val[2] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::greater<unsigned int>());

  val[3] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::bit_and<unsigned int>());

  val[4] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::bit_xor<unsigned int>());

  val[5] = cg::exclusive_scan(tile32, tile32.thread_rank(),
                              cg::bit_or<unsigned int>());

  val[6] = cg::exclusive_scan(tile32, tile32.thread_rank());
  if (tile32.thread_rank() == 31) {
    output[0] = val[0];
    output[1] = val[1];
    output[2] = val[2];
    output[3] = val[3];
    output[4] = val[4];
    output[5] = val[5];
    output[6] = val[6];
  }
}

__global__ void testIncluSive(unsigned int *output) {
  auto thread_block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(thread_block);
  unsigned int val[7];

  val[0] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::plus<unsigned int>());

  val[1] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::less<unsigned int>());

  val[2] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::greater<unsigned int>());

  val[3] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::bit_and<unsigned int>());

  val[4] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::bit_xor<unsigned int>());

  val[5] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::bit_or<unsigned int>());

  val[6] = cg::inclusive_scan(tile, tile.thread_rank());
  if (tile.thread_rank() == 31) {
    output[0] = val[0];
    output[1] = val[1];
    output[2] = val[2];
    output[3] = val[3];
    output[4] = val[4];
    output[5] = val[5];
    output[6] = val[6];
  }
}

__global__ void test(unsigned int *output) {
  auto thread_block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(thread_block);
  unsigned int val[7];
  val[0] =
      cg::inclusive_scan(tile, tile.thread_rank(), cg::plus<unsigned int>());
  printf("%d\n", val[0]);
}

int main() {
  bool checker1 = test_thread_num();
  bool checker2 = true;
  bool checker3 = true;
  unsigned int expected1[7] = {496, 0, 31, 0, 0, 31, 496};
  unsigned int expected2[7] = {465, 0, 30, 0, 31, 31, 465};

  unsigned int *output;
  unsigned int *output2;
  unsigned int *host_output = (unsigned int *)malloc(7 * sizeof(unsigned int));
  unsigned int *host_output2 = (unsigned int *)malloc(7 * sizeof(unsigned int));
  cudaMalloc((void **)&output, sizeof(unsigned int) * 7);
  cudaMalloc((void **)&output2, sizeof(unsigned int) * 7);
  testIncluSive<<<1, 256>>>(output);
  testExclusive<<<1, 256>>>(output2);
  cudaMemcpy(host_output, output, sizeof(unsigned int) * 7,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_output2, output2, sizeof(unsigned int) * 7,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 7; i++) {
    if (host_output[i] != expected1[i])
      checker2 = false;
  }
  for (int i = 0; i < 7; i++) {
    std::cout << host_output2[i] << std::endl;
    if (host_output2[i] != expected2[i])
      checker3 = false;
  }

  if (checker1 != true) {
    std::cout << "Test Thread num failed" << std::endl;
  }
  if (checker2 != true) {
    std::cout << "Test Inclusive scan failed" << std::endl;
  }
  if (checker2 != true) {
    std::cout << "Test Exclusive scan failed" << std::endl;
  }

  if (checker1 && checker2 && checker3)
    return 0;
  return -1;
}
