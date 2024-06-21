// ===------- blas_utils_matrix_mem_copy_test.cpp ---------- *- C++ -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

const std::int64_t M = 3;
const std::int64_t N = 2;

void matrix_mem_copy_test_1() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  float *devPtrA;
  devPtrA = (float *)sycl::malloc_device(M * N * sizeof(float), q_ct1);
  float host_a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float host_b[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  float host_c[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  dpct::blas::matrix_mem_copy(devPtrA, host_a, M, M, M, N, sizeof(float),
                              dpct::automatic, q_ct1, true);
  dpct::blas::matrix_mem_copy(host_b, devPtrA, M, M, M, N, sizeof(float));

  for (int i = 0; i < M * N; i++) {
    if (fabs(host_b[i] - host_c[i]) > 1e-5) {
      printf("matrix_mem_copy_test_1.1 failed\n");
      exit(-1);
    }
  }

  // Because to_ld == from_ld, matrix_mem_copy just do one copy.
  // All padding data is also copied except the last padding.
  float host_d[6] = {-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f};
  float host_e[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, -2.0f};
  dpct::blas::matrix_mem_copy(host_d, devPtrA, M /*to_ld*/, M /*from_ld*/,
                              M - 1 /*rows*/, N /*cols*/, sizeof(float));

  for (int i = 0; i < M * N; i++) {
    if (fabs(host_d[i] - host_e[i]) > 1e-5) {
      printf("matrix_mem_copy_test_1.2 failed\n");
      exit(-1);
    }
  }

  sycl::free(devPtrA, q_ct1);
}

void matrix_mem_copy_test_2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  float *devPtrA;
  devPtrA = (float *)sycl::malloc_device(M * N * sizeof(float), q_ct1);
  float host_a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float host_b[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  float host_c[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  dpct::blas::matrix_mem_copy(devPtrA, host_a, M, M, M, N, sizeof(float),
                              dpct::automatic, q_ct1, true);
  dpct::blas::matrix_mem_copy(host_b, devPtrA, M, M, M, N, sizeof(float));

  for (int i = 0; i < M * N; i++) {
    if (fabs(host_b[i] - host_c[i]) > 1e-5) {
      printf("matrix_mem_copy_test_2.1 failed\n");
      exit(-1);
    }
  }

  float host_d[4] = {-2.0f, -2.0f, -2.0f, -2.0f};
  float host_e[4] = {1.0f, 2.0f, 4.0f, 5.0f};
  dpct::blas::matrix_mem_copy(host_d, devPtrA, M - 1 /*to_ld*/, M /*from_ld*/,
                              M - 1 /*rows*/, N /*cols*/, sizeof(float));

  for (int i = 0; i < (M - 1) * N; i++) {
    if (fabs(host_d[i] - host_e[i]) > 1e-5) {
      printf("matrix_mem_copy_test_2.2 failed\n");
      exit(-1);
    }
  }

  sycl::free(devPtrA, q_ct1);
}

int main() {
  matrix_mem_copy_test_1();
  matrix_mem_copy_test_2();
  printf("matrix_mem_copy_test passed\n");
  return 0;
}
