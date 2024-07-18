// ===------ blas_utils_gels-usm.cpp ---------------------- *- C++ -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

#include <cmath>
#include <cstdio>

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float A[9] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
  float B[9] = {1, 2, 3, 4, 5, 6, 7, 9, 9};

  float *A_dev_mem;
  float *B_dev_mem;
  A_dev_mem = sycl::malloc_device<float>(18, q_ct1);
  B_dev_mem = sycl::malloc_device<float>(18, q_ct1);
  q_ct1.memcpy(A_dev_mem, A, sizeof(float) * 9);
  q_ct1.memcpy(A_dev_mem + 9, A, sizeof(float) * 9);
  q_ct1.memcpy(B_dev_mem, B, sizeof(float) * 9);
  q_ct1.memcpy(B_dev_mem + 9, B, sizeof(float) * 9).wait();

  float **As;
  float **Bs;
  As = sycl::malloc_device<float *>(2, q_ct1);
  Bs = sycl::malloc_device<float *>(2, q_ct1);

  q_ct1.memcpy(As, &A_dev_mem, sizeof(float *));
  float *temp_a = A_dev_mem + 9;
  q_ct1.memcpy(As + 1, &temp_a, sizeof(float *));
  q_ct1.memcpy(Bs, &B_dev_mem, sizeof(float *));
  float *temp_b = B_dev_mem + 9;
  q_ct1.memcpy(Bs + 1, &temp_b, sizeof(float *)).wait();

  dpct::blas::descriptor_ptr handle;
  handle = new dpct::blas::descriptor();

  int info;
  dpct::blas::gels_batch_wrapper(handle, oneapi::mkl::transpose::nontrans, 3, 3,
                                 3, As, 3, Bs, 3, &info, NULL, 2);
  q_ct1.wait();

  float A_host_mem[18];
  float B_host_mem[18];
  q_ct1.memcpy(A_host_mem, A_dev_mem, sizeof(float) * 18);
  q_ct1.memcpy(B_host_mem, B_dev_mem, sizeof(float) * 18).wait();

  printf("a:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", A_host_mem[i]);
  }
  printf("\n");
  printf("b:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", B_host_mem[i]);
  }
  printf("\n");

  float A_ref[18] = {-6.164414,  0.367448,   0.612414,   -18.168798, -2.982405,
                     -0.509851,  -33.417614, -6.653060,  -4.242642,  -6.164414,
                     0.367448,   0.612414,   -18.168798, -2.982405,  -0.509851,
                     -33.417614, -6.653060,  -4.242642};
  float B_ref[18] = {0.461538,  0.166667,  -0.064103, 0.000000, 0.166667,
                     0.166667,  -1.230769, 0.666667,  0.282051, 0.461538,
                     0.166667,  -0.064103, 0.000000,  0.166667, 0.166667,
                     -1.230769, 0.666667,  0.282051};

  bool pass = true;
  for (int i = 0; i < 18; i++) {
    if (std::fabs(A_ref[i] - A_host_mem[i]) > 0.01) {
      pass = false;
      break;
    }
    if (std::fabs(B_ref[i] - B_host_mem[i]) > 0.01) {
      pass = false;
      break;
    }
  }

  if (pass) {
    printf("pass\n");
    return 0;
  }
  printf("fail\n");
  return -1;
}
