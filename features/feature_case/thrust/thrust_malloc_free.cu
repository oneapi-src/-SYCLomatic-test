// ====------ thrust_malloc_free.cu--------------- *- CUDA -* --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-----------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>

int main() {

  {
    const int N = 100;
    thrust::device_system_tag device_sys;
    thrust::pointer<int, thrust::device_system_tag> ptr =
        thrust::malloc<int>(device_sys, N);

    // deallocate ptr with thrust::free
    thrust::free(device_sys, ptr);
  }

  {

    // allocate some memory with thrust::malloc
    const int N = 100;
    thrust::device_system_tag device_sys;
    thrust::pointer<void, thrust::device_system_tag> void_ptr =
        thrust::malloc(device_sys, N);
    // manipulate memory

    // deallocate void_ptr with thrust::free
    thrust::free(device_sys, void_ptr);
  }

  return 0;
}
