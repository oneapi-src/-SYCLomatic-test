// ====------ asm_fence.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

inline void fence(int *arr, int *lock, bool reset,
                  const sycl::nd_item<3> &item_ct1) {

  if (item_ct1.get_local_id(2) < 10) {
    arr[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) * 2;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (item_ct1.get_local_id(2) == 0) {

    if(reset){
      lock[0] = 0;
      return;
    }
    int val = 1;
    
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

    lock[0] = val;
  }
  
}

void kernel(int *arr, int *lock, bool reset, const sycl::nd_item<3> &item_ct1) {
  fence(arr, lock, reset, item_ct1);
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  int *lock;
  lock = sycl::malloc_shared<int>(1, q_ct1);
  lock[0] = 1;
  
  int *arr, *brr;
  arr = sycl::malloc_shared<int>(10, q_ct1);
  q_ct1.memset(arr, 0, sizeof(int) * 10).wait();

  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 10), sycl::range<3>(1, 1, 10)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel(arr, lock, false, item_ct1);
      });

  dev_ct1.queues_wait_and_throw();

  int res = 0;
  for (int i = 0; i < 10; ++i) {
    if (arr[i] !=  i*2){
        res = 1;
    }
  }
  dpct::dpct_free(arr, q_ct1);
  return res;
}
