// ===------------ transform.cu -------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_gemm_utils.hpp>
#include <cstring>
#include <dpct/lib_common_utils.hpp>


void transform(dpct::blas_gemm::experimental::descriptor_ptr ltHandle, void *in, int ld_in,
               dpct::blas_gemm::experimental::order_t order_in, void *out, int ld_out,
               dpct::blas_gemm::experimental::order_t order_out, int dim1, int dim2) {
  dpct::blas_gemm::experimental::matrix_layout_ptr in_desc = NULL, out_desc = NULL;
  dpct::blas_gemm::experimental::transform_desc_ptr transform_desc = NULL;

  in_desc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, dim1, dim2, ld_in);
  out_desc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, dim1, dim2, ld_out);

  in_desc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &order_in);
  out_desc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &order_out);

  transform_desc = new dpct::blas_gemm::experimental::transform_desc_t(dpct::library_data_t::real_float);

  float alpha = 1.0f, beta = 0.0f;
  dpct::blas_gemm::experimental::matrix_transform(transform_desc, &alpha, in, in_desc, &beta, NULL, NULL, out, out_desc, 0);

  delete (in_desc);
  delete (out_desc);
  delete (transform_desc);
}

bool test_ROW() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const constexpr int m = 2;
  const constexpr int n = 33;
  const constexpr int in_ld = 4;
  void *in_dev;
  in_dev = (void *)sycl::malloc_device(n * in_ld * sizeof(int8_t), q_ct1);

  int8_t in_host[n * in_ld];
  int8_t value = 0;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % 4 < 2) {
      in_host[i] = value;
      value++;
    } else
      in_host[i] = 99;
  }
  int8_t ref_2nd[n * in_ld];
  std::memcpy(ref_2nd, in_host, n * in_ld * sizeof(int8_t));

  q_ct1.memcpy(in_dev, in_host, n * in_ld * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();

  void *out_dev;
  const constexpr int out_ld = 36;
  out_dev = (void *)sycl::malloc_device(out_ld * m * sizeof(int8_t), q_ct1);
  q_ct1.memset(out_dev, 0, out_ld * m * sizeof(int8_t)).wait();
  transform(ltHandle, in_dev, in_ld, dpct::blas_gemm::experimental::order_t::col, out_dev, out_ld,
            dpct::blas_gemm::experimental::order_t::row, m, n);

  int8_t out_host[out_ld * m];
  q_ct1.memcpy(out_host, out_dev, out_ld * m * sizeof(int8_t)).wait();

  bool pass_1st = true;
  int8_t ref_1st[out_ld * m] = 
     {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 0, 0, 0,
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 0, 0, 0};
  for (int i = 0; i < out_ld * m; i++) {
    if (i % out_ld < n) {
      if (out_host[i] != ref_1st[i]) {
        pass_1st = false;
        break;
      }
    }
  }

  for (int i = 0; i < out_ld * m; i++) {
    printf("%d, ", out_host[i]);
  }
  printf("\n");
  if (pass_1st) {
    printf("ROW 1st pass\n");
  } else {
    printf("ROW 1st fail\n");
  }

  q_ct1.memset(in_dev, 0, n * in_ld * sizeof(int8_t)).wait();
  std::memset(in_host, 0, n * in_ld * sizeof(int8_t));
  transform(ltHandle, out_dev, out_ld, dpct::blas_gemm::experimental::order_t::row, in_dev, in_ld,
            dpct::blas_gemm::experimental::order_t::col, m, n);
  q_ct1.memcpy(in_host, in_dev, n * in_ld * sizeof(int8_t)).wait();

  bool pass_2nd = true;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % in_ld < m) {
      if (in_host[i] != ref_2nd[i]) {
        pass_2nd = false;
        break;
      }
    }
  }

  for (int i = 0; i < n * in_ld; i++) {
    printf("%d, ", in_host[i]);
  }
  printf("\n");
  if (pass_2nd) {
    printf("ROW 2nd pass\n");
  } else {
    printf("ROW 2nd fail\n");
  }

  delete (ltHandle);

  return pass_1st && pass_2nd;
}

bool test_COL32() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const constexpr int m = 2;
  const constexpr int n = 33;
  const constexpr int in_ld = 4;
  void *in_dev;
  in_dev = (void *)sycl::malloc_device(n * in_ld * sizeof(int8_t), q_ct1);

  int8_t in_host[n * in_ld];
  int8_t value = 0;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % 4 < 2) {
      in_host[i] = value;
      value++;
    } else
      in_host[i] = 99;
  }
  int8_t ref_2nd[n * in_ld];
  std::memcpy(ref_2nd, in_host, n * in_ld * sizeof(int8_t));

  q_ct1.memcpy(in_dev, in_host, n * in_ld * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();

  void *out_dev;
  const constexpr int out_ld = 64;
  out_dev = (void *)sycl::malloc_device(out_ld * m * sizeof(int8_t), q_ct1);
  q_ct1.memset(out_dev, 0, out_ld * m * sizeof(int8_t)).wait();
  transform(ltHandle, in_dev, in_ld, dpct::blas_gemm::experimental::order_t::col, out_dev, out_ld,
            dpct::blas_gemm::experimental::order_t::col32, m, n);

  int8_t out_host[out_ld * m];
  q_ct1.memcpy(out_host, out_dev, out_ld * m * sizeof(int8_t)).wait();

  bool pass_1st = true;
  int8_t ref_1st[out_ld * m] = 
     {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < out_ld * m; i++) {
    if (i % out_ld < n) {
      if (out_host[i] != ref_1st[i]) {
        pass_1st = false;
        break;
      }
    }
  }

  for (int i = 0; i < out_ld * m; i++) {
    printf("%d, ", out_host[i]);
  }
  printf("\n");
  if (pass_1st) {
    printf("COL32 1st pass\n");
  } else {
    printf("COL32 1st fail\n");
  }

  q_ct1.memset(in_dev, 0, n * in_ld * sizeof(int8_t)).wait();
  std::memset(in_host, 0, n * in_ld * sizeof(int8_t));
  transform(ltHandle, out_dev, out_ld, dpct::blas_gemm::experimental::order_t::col32, in_dev, in_ld,
            dpct::blas_gemm::experimental::order_t::col, m, n);
  q_ct1.memcpy(in_host, in_dev, n * in_ld * sizeof(int8_t)).wait();

  bool pass_2nd = true;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % in_ld < m) {
      if (in_host[i] != ref_2nd[i]) {
        pass_2nd = false;
        break;
      }
    }
  }

  for (int i = 0; i < n * in_ld; i++) {
    printf("%d, ", in_host[i]);
  }
  printf("\n");
  if (pass_2nd) {
    printf("COL32 2nd pass\n");
  } else {
    printf("COL32 2nd fail\n");
  }

  delete (ltHandle);

  return pass_1st && pass_2nd;
}

bool test_COL4_4R2_8C() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const constexpr int m = 2;
  const constexpr int n = 33;
  const constexpr int in_ld = 4;
  void *in_dev;
  in_dev = (void *)sycl::malloc_device(n * in_ld * sizeof(int8_t), q_ct1);

  int8_t in_host[n * in_ld];
  int8_t value = 0;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % 4 < 2) {
      in_host[i] = value;
      value++;
    } else
      in_host[i] = 99;
  }
  int8_t ref_2nd[n * in_ld];
  std::memcpy(ref_2nd, in_host, n * in_ld * sizeof(int8_t));

  q_ct1.memcpy(in_dev, in_host, n * in_ld * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();

  void *out_dev;
  const constexpr int out_ld = (32 * 8) * 2;
  out_dev = (void *)sycl::malloc_device(out_ld * m * sizeof(int8_t), q_ct1);
  q_ct1.memset(out_dev, 0, out_ld * m * sizeof(int8_t)).wait();
  transform(ltHandle, in_dev, in_ld, dpct::blas_gemm::experimental::order_t::col, out_dev, out_ld,
            dpct::blas_gemm::experimental::order_t::col4_4r2_8c, m, n);

  int8_t out_host[out_ld * m];
  q_ct1.memcpy(out_host, out_dev, out_ld * m * sizeof(int8_t)).wait();

  bool pass_1st = true;
  int8_t ref_1st[out_ld * m] = 
     {0,  2,  4,  6,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      8,  10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      16, 18, 20, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      24, 26, 28, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      32, 34, 36, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      40, 42, 44, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      48, 50, 52, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      56, 58, 60, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1,  3,  5,  7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      9,  11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      17, 19, 21, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      25, 27, 29, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      33, 35, 37, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      41, 43, 45, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      49, 51, 53, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      57, 59, 61, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      64, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      65, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < out_ld * m; i++) {
    if (i % out_ld < n) {
      if (out_host[i] != ref_1st[i]) {
        pass_1st = false;
        break;
      }
    }
  }

  for (int i = 0; i < out_ld * m; i++) {
    printf("%d, ", out_host[i]);
  }
  printf("\n");
  if (pass_1st) {
    printf("COL4_4R2_8C 1st pass\n");
  } else {
    printf("COL4_4R2_8C 1st fail\n");
  }

  q_ct1.memset(in_dev, 0, n * in_ld * sizeof(int8_t)).wait();
  std::memset(in_host, 0, n * in_ld * sizeof(int8_t));
  transform(ltHandle, out_dev, out_ld, dpct::blas_gemm::experimental::order_t::col4_4r2_8c, in_dev,
            in_ld, dpct::blas_gemm::experimental::order_t::col, m, n);
  q_ct1.memcpy(in_host, in_dev, n * in_ld * sizeof(int8_t)).wait();

  bool pass_2nd = true;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % in_ld < m) {
      if (in_host[i] != ref_2nd[i]) {
        pass_2nd = false;
        break;
      }
    }
  }

  for (int i = 0; i < n * in_ld; i++) {
    printf("%d, ", in_host[i]);
  }
  printf("\n");
  if (pass_2nd) {
    printf("COL4_4R2_8C 2nd pass\n");
  } else {
    printf("COL4_4R2_8C 2nd fail\n");
  }

  delete (ltHandle);

  return pass_1st && pass_2nd;
}

bool test_COL32_2R_4R4() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const constexpr int m = 2;
  const constexpr int n = 33;
  const constexpr int in_ld = 4;
  void *in_dev;
  in_dev = (void *)sycl::malloc_device(n * in_ld * sizeof(int8_t), q_ct1);

  int8_t in_host[n * in_ld];
  int8_t value = 0;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % 4 < 2) {
      in_host[i] = value;
      value++;
    } else
      in_host[i] = 99;
  }
  int8_t ref_2nd[n * in_ld];
  std::memcpy(ref_2nd, in_host, n * in_ld * sizeof(int8_t));

  q_ct1.memcpy(in_dev, in_host, n * in_ld * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();

  void *out_dev;
  const constexpr int out_ld = (32 * 32) * 2;
  out_dev = (void *)sycl::malloc_device(out_ld * m * sizeof(int8_t), q_ct1);
  q_ct1.memset(out_dev, 0, out_ld * m * sizeof(int8_t)).wait();
  transform(ltHandle, in_dev, in_ld, dpct::blas_gemm::experimental::order_t::col, out_dev, out_ld,
            dpct::blas_gemm::experimental::order_t::col32_2r_4r4, m, n);

  int8_t out_host[out_ld * m];
  q_ct1.memcpy(out_host, out_dev, out_ld * m * sizeof(int8_t)).wait();

  bool pass_1st = true;
  int8_t ref_1st[out_ld * m] = 
     {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < out_ld * m; i++) {
    if (i % out_ld < n) {
      if (out_host[i] != ref_1st[i]) {
        pass_1st = false;
        break;
      }
    }
  }

  for (int i = 0; i < out_ld * m; i++) {
    printf("%d, ", out_host[i]);
  }
  printf("\n");
  if (pass_1st) {
    printf("COL32_2R_4R4 1st pass\n");
  } else {
    printf("COL32_2R_4R4 1st fail\n");
  }

  q_ct1.memset(in_dev, 0, n * in_ld * sizeof(int8_t)).wait();
  std::memset(in_host, 0, n * in_ld * sizeof(int8_t));
  transform(ltHandle, out_dev, out_ld, dpct::blas_gemm::experimental::order_t::col32_2r_4r4, in_dev,
            in_ld, dpct::blas_gemm::experimental::order_t::col, m, n);
  q_ct1.memcpy(in_host, in_dev, n * in_ld * sizeof(int8_t)).wait();

  bool pass_2nd = true;
  for (int i = 0; i < n * in_ld; i++) {
    if (i % in_ld < m) {
      if (in_host[i] != ref_2nd[i]) {
        pass_2nd = false;
        break;
      }
    }
  }

  for (int i = 0; i < n * in_ld; i++) {
    printf("%d, ", in_host[i]);
  }
  printf("\n");
  if (pass_2nd) {
    printf("COL32_2R_4R4 2nd pass\n");
  } else {
    printf("COL32_2R_4R4 2nd fail\n");
  }

  delete (ltHandle);

  return pass_1st && pass_2nd;
}

// Input col_major matrix:
// 2 rows * 33 columns, ld is 4
int main() {
  bool pass = true;
  pass = test_ROW() && pass;
  pass = test_COL32() && pass;
  pass = test_COL4_4R2_8C() && pass;
  pass = test_COL32_2R_4R4() && pass;
  return pass ? 0 : 1;
}
