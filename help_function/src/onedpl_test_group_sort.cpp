// ====------ onedpl_test_group_sort.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>

template <int GROUP_THREADS, typename InputT, int ITEMS_PER_THREAD,
          typename InputIteratorT>
void load_striped(int linear_tid, InputIteratorT block_itr,
                       InputT (&items)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = block_itr[linear_tid + ITEM * GROUP_THREADS];
  }
}

template <int GROUP_THREADS, typename T, int ITEMS_PER_THREAD,
          typename OutputIteratorT>
void store_striped(int linear_tid, OutputIteratorT block_itr,
                        T (&items)[ITEMS_PER_THREAD]) {
  OutputIteratorT thread_itr = block_itr + linear_tid;
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_itr[(ITEM * GROUP_THREADS)] = items[ITEM];
  }
}

bool helper_function(const int* ptr,const char* func_name){
  int expected[512];
  for (int i = 0; i < 512; ++i) {
      expected[i] = (i % 128) * 4 + (i / 128);
  }
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != expected[i]) {
      std::cout << func_name <<" failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << func_name <<" pass\n";
  return true;
}

bool test_sort_blocked() {
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 128; i++) {
    data[4 * i + 0] = i;
    data[4 * i + 1] = i + 1 * 128;
    data[4 * i + 2] = i + 2 * 128;
    data[4 * i + 3] = i + 3 * 128;
  }

  sycl::buffer<int, 1> buffer(data, 512);
  q.submit([&](sycl::handler &h) {
    using group_sort = dpct::group::radix_sort<int, 4>;
    size_t temp_storage_size = group_sort::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(
        sycl::range<1>(temp_storage_size), h);
    sycl::accessor dacc(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          load_striped<128>(item.get_local_linear_id(), d, thread_data);
          group_sort(tmp).sort_blocked(item, thread_data);
          store_striped<128>(item.get_local_linear_id(), d, thread_data);
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_function(ptr,"test_sort_blocked");
}

bool test_sort_blocked_to_striped() {
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 512; i++) data[i] = i;

  sycl::buffer<int, 1> buffer(data, 512);

  q.submit([&](sycl::handler &h) {
    using group_sort = dpct::group::radix_sort<int, 4>;
    size_t temp_storage_size = group_sort::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(
        sycl::range<1>(temp_storage_size), h);
    sycl::accessor data_accessor(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = data_accessor.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          load_striped<128>(item.get_local_linear_id(), d, thread_data);
          group_sort(tmp).sort_blocked_to_striped(item, thread_data);
          store_striped<128>(item.get_local_linear_id(), d, thread_data);
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_function(ptr,"test_sort_blocked_to_striped");
}

int main() {
  return !(test_sort_blocked() && test_sort_blocked_to_striped());
}

