// ====------ cub_device_partition.cu ---------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cub/cub.cuh>

#include <algorithm>
#include <initializer_list>
#include <iostream>

struct LessThan {
  int compare;
  explicit LessThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const { return (a < compare); }
};

struct GreaterThan {
  int compare;
  explicit GreaterThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const { return a > compare; }
};

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMallocManaged(&p, sizeof(T) * list.size());
  std::copy(list.begin(), list.end(), p);
  return p;
}

int test() {

  // Flagged
  {
    int num_items = 8;
    int *d_in = init({1, 2, 3, 4, 5, 6, 7, 8});
    int *d_flags = init({1, 0, 0, 1, 0, 1, 1, 0});
    int *d_out = init({0, 0, 0, 0, 0, 0, 0, 0});
    int *d_num_selected_out = init({0});

    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in,
                                  d_flags, d_out, d_num_selected_out,
                                  num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in,
                                  d_flags, d_out, d_num_selected_out,
                                  num_items);
    // d_out                 <-- [1, 4, 6, 7, 8, 5, 3, 2]
    // d_num_selected_out    <-- [4]
    cudaDeviceSynchronize();
    if (*d_num_selected_out != 4)
      return 1;
    int expected[] = {1, 4, 6, 7, 8, 5, 3, 2};
    if (!std::equal(d_out, d_out + num_items, expected))
      return 2;
  }

  // If
  {
    int num_items = 8;
    int *d_in = init({0, 2, 3, 9, 5, 2, 81, 8});
    int *d_out = init({0, 0, 0, 0, 0, 0, 0, 0});
    int *d_num_selected_out = init({0});
    LessThan select_op(7);

    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                             d_num_selected_out, num_items, select_op);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                             d_num_selected_out, num_items, select_op);
    // d_out                 <-- [0, 2, 3, 5, 2, 8, 81, 9]
    // d_num_selected_out    <-- [5]
    cudaDeviceSynchronize();
    if (*d_num_selected_out != 5)
      return 3;
    int expected[] = {0, 2, 3, 5, 2, 8, 81, 9};
    if (!std::equal(d_out, d_out + num_items, expected))
      return 4;
  }

  // If && custom select op
  {
    int num_items = 8;
    int *d_in = init({0, 2, 3, 9, 5, 2, 81, 8});
    int *d_large_out = init({0, 0, 0, 0, 0, 0, 0, 0});
    int *d_small_out = init({0, 0, 0, 0, 0, 0, 0, 0});
    int *d_unselected_out = init({0, 0, 0, 0, 0, 0, 0, 0});
    int *d_num_selected_out = init({0, 0});

    LessThan small_items_selector(7);
    GreaterThan large_items_selector(50);

    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in,
                             d_large_out, d_small_out, d_unselected_out,
                             d_num_selected_out, num_items,
                             large_items_selector, small_items_selector);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in,
                             d_large_out, d_small_out, d_unselected_out,
                             d_num_selected_out, num_items,
                             large_items_selector, small_items_selector);
    // d_large_out                 <-- [ 81,  ,  ,  ,  ,  , 0, 0 ]
    // d_unselected_out            <-- [  9, 8,  ,  ,  ,  ,  ,   ]
    // d_small_out                 <-- [  0, 2, 3, 5, 2,  ,  ,   ]
    // d_num_selected_out          <-- [  1, 5 ]
    cudaDeviceSynchronize();
    if (d_num_selected_out[0] != 1 && d_num_selected_out[1] != 5)
      return 5;
    if (d_large_out[0] != 81)
      return 6;
    if (d_unselected_out[0] != 9 && d_unselected_out[1] != 8)
      return 7;
    int expected_small_out[] = {0, 2, 3, 5, 2};
    if (!std::equal(d_small_out, d_small_out + 5, expected_small_out))
      return 8;
  }

  return 0;
}

int main() { return test(); }
