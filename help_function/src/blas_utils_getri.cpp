// ====------ blas_utils_getri.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

template<class T>
bool verify_data(T* data, T* expect, int num) {
  for(int i = 0; i < num; ++i) {
    if(std::abs(data[i] - expect[i]) > 0.01) {
      return false;
    }
  }
  return true;
}

template<class T>
int test() {
  int n = 2;
  T *A = (T *)malloc(n * n * sizeof(T));
  A[0] = 2;
  A[1] = 0.5;
  A[2] = 4;
  A[3] = 1;

  int *Pivots_h = (int *)malloc(2 * n * sizeof(int));
  Pivots_h[0] = 2;
  Pivots_h[1] = 2;
  Pivots_h[2] = 2;
  Pivots_h[3] = 2;

  sycl::queue *handle;
  handle = &dpct::get_default_queue();
  std::cout << "Device Name: " << handle->get_device().get_info<sycl::info::device::name>() << std::endl;

  T **Aarray;
  T **Carray;
  T *a0, *a1;
  T *c0, *c1;
  int *Pivots;
  int *dInfo;
  size_t sizeA = n * n * sizeof(T);

  Aarray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  Carray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  a0 = (T *)dpct::dpct_malloc(sizeA);
  c0 = (T *)dpct::dpct_malloc(sizeA);
  a1 = (T *)dpct::dpct_malloc(sizeA);
  c1 = (T *)dpct::dpct_malloc(sizeA);
  Pivots = (int *)dpct::dpct_malloc(2 * n * sizeof(int));
  dInfo = (int *)dpct::dpct_malloc(2 * sizeof(int));

  dpct::dpct_memcpy(Pivots, Pivots_h, 2 * n * sizeof(int),
                    dpct::host_to_device);
  dpct::dpct_memcpy(a0, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(a1, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(Aarray, &a0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Carray, &c0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Aarray + 1, &a1, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Carray + 1, &c1, sizeof(T *), dpct::host_to_device);

  dpct::getri_batch_wrapper(*handle, n, (const T **)Aarray, n, Pivots,
                            Carray, n, dInfo, 2);
  dpct::get_current_device().queues_wait_and_throw();

  T *inv = (T *)malloc(2 * sizeA);

  dpct::dpct_memcpy(inv, c0, sizeA, dpct::device_to_host);
  dpct::dpct_memcpy(inv + n * n, c1, sizeA, dpct::device_to_host);

  dpct::dpct_free(Aarray);
  dpct::dpct_free(Carray);
  dpct::dpct_free(a0);
  dpct::dpct_free(c0);
  dpct::dpct_free(Pivots);
  dpct::dpct_free(dInfo);
  dpct::dpct_free(a1);
  dpct::dpct_free(c1);

  handle = nullptr;

  printf("inv0[0]:%f, inv0[1]:%f, inv0[2]:%f, inv0[3]:%f\n", inv[0], inv[1], inv[2],
         inv[3]);
  printf("inv1[0]:%f, inv1[1]:%f, inv1[2]:%f, inv1[3]:%f\n", inv[4], inv[5], inv[6],
         inv[7]);

  // check result:
  T expect[8] = {
    -2, 1, 1.5, -0.5,
    -2, 1, 1.5, -0.5
  };

  bool success = false;
  if(verify_data(inv, expect, 8))
    success = true;

  free(A);
  free(inv);
  printf("done.\n");

  return (success ? 0 : 1);
}
int main() {
  bool pass = true;
  if(test<float>()) {
    pass = false;
    printf("float fail\n");
  }
  if(test<double>()) {
    pass = false;
    printf("double fail\n");
  }
  return (pass ? 0 : 1);
}
