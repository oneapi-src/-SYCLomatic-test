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

bool all_pass = true;

void test1() {
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
    printf("test1 pass\n");
    return;
  }
  printf("test1 fail\n");
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
  all_pass = false;
}

void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  double A[9] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
  double B[9] = {1, 2, 3, 4, 5, 6, 7, 9, 9};

  double *A_dev_mem;
  double *B_dev_mem;
  A_dev_mem = sycl::malloc_device<double>(18, q_ct1);
  B_dev_mem = sycl::malloc_device<double>(18, q_ct1);
  q_ct1.memcpy(A_dev_mem, A, sizeof(double) * 9);
  q_ct1.memcpy(A_dev_mem + 9, A, sizeof(double) * 9);
  q_ct1.memcpy(B_dev_mem, B, sizeof(double) * 9);
  q_ct1.memcpy(B_dev_mem + 9, B, sizeof(double) * 9).wait();

  double **As;
  double **Bs;
  As = sycl::malloc_device<double *>(2, q_ct1);
  Bs = sycl::malloc_device<double *>(2, q_ct1);

  q_ct1.memcpy(As, &A_dev_mem, sizeof(double *));
  double *temp_a = A_dev_mem + 9;
  q_ct1.memcpy(As + 1, &temp_a, sizeof(double *));
  q_ct1.memcpy(Bs, &B_dev_mem, sizeof(double *));
  double *temp_b = B_dev_mem + 9;
  q_ct1.memcpy(Bs + 1, &temp_b, sizeof(double *)).wait();

  dpct::blas::descriptor_ptr handle;
  handle = new dpct::blas::descriptor();

  int info;
  dpct::blas::gels_batch_wrapper(handle, oneapi::mkl::transpose::nontrans, 3, 3,
                                 3, As, 3, Bs, 3, &info, NULL, 2);
  q_ct1.wait();

  double A_host_mem[18];
  double B_host_mem[18];
  q_ct1.memcpy(A_host_mem, A_dev_mem, sizeof(double) * 18);
  q_ct1.memcpy(B_host_mem, B_dev_mem, sizeof(double) * 18).wait();

  double A_ref[18] = {-6.164414,  0.367448,   0.612414,   -18.168798, -2.982405,
                      -0.509851,  -33.417614, -6.653060,  -4.242642,  -6.164414,
                      0.367448,   0.612414,   -18.168798, -2.982405,  -0.509851,
                      -33.417614, -6.653060,  -4.242642};
  double B_ref[18] = {0.461538,  0.166667,  -0.064103, 0.000000, 0.166667,
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
    printf("test2 pass\n");
    return;
  }
  printf("test2 fail\n");
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
  all_pass = false;
}

void test3() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::float2 A[9] = {
      sycl::float2(2, 0),  sycl::float2(3, 0),  sycl::float2(5, 0),
      sycl::float2(7, 0),  sycl::float2(11, 0), sycl::float2(13, 0),
      sycl::float2(17, 0), sycl::float2(19, 0), sycl::float2(23, 0)};
  sycl::float2 B[9] = {
      sycl::float2(1, 0), sycl::float2(2, 0), sycl::float2(3, 0),
      sycl::float2(4, 0), sycl::float2(5, 0), sycl::float2(6, 0),
      sycl::float2(7, 0), sycl::float2(9, 0), sycl::float2(9, 0)};

  sycl::float2 *A_dev_mem;
  sycl::float2 *B_dev_mem;
  A_dev_mem = sycl::malloc_device<sycl::float2>(18, q_ct1);
  B_dev_mem = sycl::malloc_device<sycl::float2>(18, q_ct1);
  q_ct1.memcpy(A_dev_mem, A, sizeof(sycl::float2) * 9);
  q_ct1.memcpy(A_dev_mem + 9, A, sizeof(sycl::float2) * 9);
  q_ct1.memcpy(B_dev_mem, B, sizeof(sycl::float2) * 9);
  q_ct1.memcpy(B_dev_mem + 9, B, sizeof(sycl::float2) * 9).wait();

  sycl::float2 **As;
  sycl::float2 **Bs;
  As = sycl::malloc_device<sycl::float2 *>(2, q_ct1);
  Bs = sycl::malloc_device<sycl::float2 *>(2, q_ct1);

  q_ct1.memcpy(As, &A_dev_mem, sizeof(sycl::float2 *));
  sycl::float2 *temp_a = A_dev_mem + 9;
  q_ct1.memcpy(As + 1, &temp_a, sizeof(sycl::float2 *));
  q_ct1.memcpy(Bs, &B_dev_mem, sizeof(sycl::float2 *));
  sycl::float2 *temp_b = B_dev_mem + 9;
  q_ct1.memcpy(Bs + 1, &temp_b, sizeof(sycl::float2 *)).wait();

  dpct::blas::descriptor_ptr handle;
  handle = new dpct::blas::descriptor();

  int info;
  dpct::blas::gels_batch_wrapper(handle, oneapi::mkl::transpose::nontrans, 3, 3,
                                 3, As, 3, Bs, 3, &info, NULL, 2);
  q_ct1.wait();

  sycl::float2 A_host_mem[18];
  sycl::float2 B_host_mem[18];
  q_ct1.memcpy(A_host_mem, A_dev_mem, sizeof(sycl::float2) * 18);
  q_ct1.memcpy(B_host_mem, B_dev_mem, sizeof(sycl::float2) * 18).wait();

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
    if (std::fabs(A_ref[i] - A_host_mem[i].x()) > 0.01) {
      pass = false;
      break;
    }
    if (std::fabs(B_ref[i] - B_host_mem[i].x()) > 0.01) {
      pass = false;
      break;
    }
  }

  if (pass) {
    printf("test3 pass\n");
    return;
  }
  printf("test3 fail\n");
  printf("a:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", A_host_mem[i].x());
  }
  printf("\n");
  printf("b:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", B_host_mem[i].x());
  }
  printf("\n");
  all_pass = false;
}

void test4() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::double2 A[9] = {
      sycl::double2(2, 0),  sycl::double2(3, 0),  sycl::double2(5, 0),
      sycl::double2(7, 0),  sycl::double2(11, 0), sycl::double2(13, 0),
      sycl::double2(17, 0), sycl::double2(19, 0), sycl::double2(23, 0)};
  sycl::double2 B[9] = {
      sycl::double2(1, 0), sycl::double2(2, 0), sycl::double2(3, 0),
      sycl::double2(4, 0), sycl::double2(5, 0), sycl::double2(6, 0),
      sycl::double2(7, 0), sycl::double2(9, 0), sycl::double2(9, 0)};

  sycl::double2 *A_dev_mem;
  sycl::double2 *B_dev_mem;
  A_dev_mem = sycl::malloc_device<sycl::double2>(18, q_ct1);
  B_dev_mem = sycl::malloc_device<sycl::double2>(18, q_ct1);
  q_ct1.memcpy(A_dev_mem, A, sizeof(sycl::double2) * 9);
  q_ct1.memcpy(A_dev_mem + 9, A, sizeof(sycl::double2) * 9);
  q_ct1.memcpy(B_dev_mem, B, sizeof(sycl::double2) * 9);
  q_ct1.memcpy(B_dev_mem + 9, B, sizeof(sycl::double2) * 9).wait();

  sycl::double2 **As;
  sycl::double2 **Bs;
  As = sycl::malloc_device<sycl::double2 *>(2, q_ct1);
  Bs = sycl::malloc_device<sycl::double2 *>(2, q_ct1);

  q_ct1.memcpy(As, &A_dev_mem, sizeof(sycl::double2 *));
  sycl::double2 *temp_a = A_dev_mem + 9;
  q_ct1.memcpy(As + 1, &temp_a, sizeof(sycl::double2 *));
  q_ct1.memcpy(Bs, &B_dev_mem, sizeof(sycl::double2 *));
  sycl::double2 *temp_b = B_dev_mem + 9;
  q_ct1.memcpy(Bs + 1, &temp_b, sizeof(sycl::double2 *)).wait();

  dpct::blas::descriptor_ptr handle;
  handle = new dpct::blas::descriptor();

  int info;
  dpct::blas::gels_batch_wrapper(handle, oneapi::mkl::transpose::nontrans, 3, 3,
                                 3, As, 3, Bs, 3, &info, NULL, 2);
  q_ct1.wait();

  sycl::double2 A_host_mem[18];
  sycl::double2 B_host_mem[18];
  q_ct1.memcpy(A_host_mem, A_dev_mem, sizeof(sycl::double2) * 18);
  q_ct1.memcpy(B_host_mem, B_dev_mem, sizeof(sycl::double2) * 18).wait();

  double A_ref[18] = {-6.164414,  0.367448,   0.612414,   -18.168798, -2.982405,
                      -0.509851,  -33.417614, -6.653060,  -4.242642,  -6.164414,
                      0.367448,   0.612414,   -18.168798, -2.982405,  -0.509851,
                      -33.417614, -6.653060,  -4.242642};
  double B_ref[18] = {0.461538,  0.166667,  -0.064103, 0.000000, 0.166667,
                      0.166667,  -1.230769, 0.666667,  0.282051, 0.461538,
                      0.166667,  -0.064103, 0.000000,  0.166667, 0.166667,
                      -1.230769, 0.666667,  0.282051};

  bool pass = true;
  for (int i = 0; i < 18; i++) {
    if (std::fabs(A_ref[i] - A_host_mem[i].x()) > 0.01) {
      pass = false;
      break;
    }
    if (std::fabs(B_ref[i] - B_host_mem[i].x()) > 0.01) {
      pass = false;
      break;
    }
  }

  if (pass) {
    printf("test4 pass\n");
    return;
  }
  printf("test4 fail\n");
  printf("a:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", A_host_mem[i].x());
  }
  printf("\n");
  printf("b:\n");
  for (int i = 0; i < 18; i++) {
    printf("%f, ", B_host_mem[i].x());
  }
  printf("\n");
  all_pass = false;
}

int main() {
  test1();
  test2();
  test3();
  test4();
  if (all_pass)
    return 0;
  return 1;
}
