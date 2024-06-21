// ====------ thrust_temporary_buffer.cu--------------- *- CUDA -* ----======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>

int main() {

  // allocate storage for 100 ints with thrust::get_temporary_buffer
  const int N = 100;
  typedef thrust::pair<thrust::pointer<int, thrust::device_system_tag>,
                       std::ptrdiff_t>
      ptr_and_size_t;
  thrust::device_system_tag device_sys;

  ptr_and_size_t ptr_and_size =
      thrust::get_temporary_buffer<int>(device_sys, N);
  // manipulate up to 100 ints
  for (int i = 0; i < ptr_and_size.second; ++i) {
    *ptr_and_size.first = i;
  }

  // deallocate storage with thrust::return_temporary_buffer
  thrust::return_temporary_buffer(device_sys, ptr_and_size.first,
                                  ptr_and_size.second);

  return 0;
}
