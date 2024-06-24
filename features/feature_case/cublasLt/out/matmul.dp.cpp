// ===------------ matmul.cu ----------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_gemm_utils.hpp>
#include <cstdint>
#include <stdexcept>
#include <dpct/lib_common_utils.hpp>


const constexpr int COL_TURING = 0;
const constexpr int COL_AMPERE = 1;

// The original source of below two functions was under the license below:
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Repo: https://github.com/TimDettmers/bitsandbytes.git
inline int checkCublasStatus(int status) {
    if (status != 0) {
        printf("cuBLAS API failed with status %d\n", status);
        //throw std::logic_error("cuBLAS API failed");
        return 1;
    }
    return 0;
}

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(dpct::blas_gemm::experimental::descriptor_ptr ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
 try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int has_error = 0;
    dpct::blas_gemm::experimental::matmul_desc_ptr matmulDesc = NULL;
    dpct::blas_gemm::experimental::matrix_layout_ptr Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    oneapi::mkl::transpose opT = oneapi::mkl::transpose::trans;
    dpct::blas_gemm::experimental::pointer_mode_t alphaVec = dpct::blas_gemm::experimental::pointer_mode_t::alpha_device_vector_beta_zero;
    dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
    dpct::blas_gemm::experimental::order_t col_turing = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
    dpct::blas_gemm::experimental::order_t col_ampere = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;

    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Adesc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda)));
    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Bdesc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb)));

    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Adesc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Bdesc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col_turing)));
    else
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Bdesc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(dpct::compute_type::i32, dpct::library_data_t::real_int32)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b, &opT)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int32, m, n, ldc)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, 0)));
    }
    else
    {
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(dpct::compute_type::i32, dpct::library_data_t::real_float)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b, &opT)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, ldc)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, 0)));
      }
      else
      {
        has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(dpct::blas_gemm::experimental::matmul_desc_t::attribute::pointer_mode, &alphaVec)));
        has_error |= checkCublasStatus(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, 0)));
      }
    }

    q_ct1.wait();

    if (Cdesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Cdesc)));
    if (Bdesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Bdesc)));
    if (Adesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Adesc)));
    if (matmulDesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (matmulDesc)));
    if(has_error == 1)
      printf("error detected");

    return has_error;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void transform(dpct::blas_gemm::experimental::descriptor_ptr ltHandle, const void *in, int ld_in,
               dpct::blas_gemm::experimental::matrix_layout_ptr layout_in, void *out, int ld_out,
               dpct::blas_gemm::experimental::matrix_layout_ptr layout_out) {
  dpct::blas_gemm::experimental::transform_desc_ptr transform_desc = NULL;
  transform_desc = new dpct::blas_gemm::experimental::transform_desc_t(dpct::library_data_t::real_float);
  float alpha = 1.0f, beta = 0.0f;
  dpct::blas_gemm::experimental::matrix_transform(transform_desc, &alpha, in, layout_in, &beta, NULL, NULL, out, layout_out, 0);
  delete (transform_desc);
}

// igemmlt<COL_TURING, 32, 0>
bool test1() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int32_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int32, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int32_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col4_4r2_8c = (int8_t *)sycl::malloc_device(((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int32_t *)sycl::malloc_device(m * 32 * sizeof(std::int32_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col4_4r2_8c = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 8 - 1) / 8) * 8 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int32, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col4_4r2_8c = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col4_4r2_8c->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col4_4r2_8c);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  // Matmul
  igemmlt<COL_TURING, 32, 0>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                             nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                             m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int32_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int32_t)).wait();

  bool error = false;
  int32_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col4_4r2_8c);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);

  return !error;
}

// igemmlt<COL_TURING, 8, 0>
bool test2() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int8_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int8_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col4_4r2_8c = (int8_t *)sycl::malloc_device(((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col4_4r2_8c = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 8 - 1) / 8) * 8 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col4_4r2_8c = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col4_4r2_8c->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col4_4r2_8c);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  // Matmul
  igemmlt<COL_TURING, 8, 0>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                            nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                            m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int8_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int8_t)).wait();

  bool error = false;
  int8_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col4_4r2_8c);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);

  return !error;
}

// igemmlt<COL_TURING, 8, 1>
bool test3() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int8_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int8_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col4_4r2_8c = (int8_t *)sycl::malloc_device(((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col4_4r2_8c = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 8 - 1) / 8) * 8 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col4_4r2_8c = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col4_4r2_8c->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col4_4r2_8c);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  float *alpha;
  alpha = sycl::malloc_shared<float>(4, q_ct1);
  alpha[0] = 0;
  alpha[1] = 1;
  alpha[2] = 2;
  alpha[3] = 3;

  // Matmul
  igemmlt<COL_TURING, 8, 1>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                            alpha, m * 32, ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int8_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int8_t)).wait();

  bool error = false;
  int8_t C_ref[m * n] = {0, 17, 40, 69, 0, 6, 16, 30};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col4_4r2_8c);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);
  dpct::dpct_free(alpha, q_ct1);

  return !error;
}

// igemmlt<COL_AMPERE, 32, 0>
bool test4() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int32_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int32, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int32_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col32_2r_4r4 = (int8_t *)sycl::malloc_device(((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int32_t *)sycl::malloc_device(m * 32 * sizeof(std::int32_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col32_2r_4r4 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 32 - 1) / 32) * 32 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int32, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col32_2r_4r4 = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col32_2r_4r4->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32_2r_4r4);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  // Matmul
  igemmlt<COL_AMPERE, 32, 0>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4,
                             C_col32, nullptr, m * 32,
                             ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int32_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int32_t)).wait();

  bool error = false;
  int32_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col32_2r_4r4);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);

  return !error;
}

// igemmlt<COL_AMPERE, 8, 0>
bool test5() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int8_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int8_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col32_2r_4r4 = (int8_t *)sycl::malloc_device(((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col32_2r_4r4 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 32 - 1) / 32) * 32 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col32_2r_4r4 = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col32_2r_4r4->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32_2r_4r4);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  // Matmul
  igemmlt<COL_AMPERE, 8, 0>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4, C_col32,
                            nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                            m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int8_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int8_t)).wait();

  bool error = false;
  int8_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col32_2r_4r4);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);

  return !error;
}

// igemmlt<COL_AMPERE, 8, 1>
bool test6() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  Adev = (void *)sycl::malloc_device(m * k * sizeof(int8_t), q_ct1);
  Bdev = (void *)sycl::malloc_device(n * k * sizeof(int8_t), q_ct1);
  Cdev = (void *)sycl::malloc_device(m * n * sizeof(int8_t), q_ct1);

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  q_ct1.memcpy(Adev, Ahost, m * k * sizeof(int8_t));
  q_ct1.memcpy(Bdev, Bhost, n * k * sizeof(int8_t)).wait();

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  Adesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, lda);
  Bdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, n, k, ldb);
  Cdesc_col_major = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, ldc);

  // Convert A and B
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int8_t *C_col32;
  A_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  B_col32_2r_4r4 = (int8_t *)sycl::malloc_device(((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t), q_ct1);
  C_col32 = (int8_t *)sycl::malloc_device(m * 32 * sizeof(std::int8_t), q_ct1);
  Adesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, k, m * 32);
  Bdesc_col32_2r_4r4 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, k, n, ((n + 32 - 1) / 32) * 32 * 32);
  Cdesc_col32 = new dpct::blas_gemm::experimental::matrix_layout_t(dpct::library_data_t::real_int8, m, n, m * 32);
  dpct::blas_gemm::experimental::order_t col32 = dpct::blas_gemm::experimental::order_t::col32;
  dpct::blas_gemm::experimental::order_t col32_2r_4r4 = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;
  Adesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);
  Bdesc_col32_2r_4r4->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32_2r_4r4);
  Cdesc_col32->set_attribute(dpct::blas_gemm::experimental::matrix_layout_t::attribute::order, &col32);

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  float *alpha;
  alpha = sycl::malloc_shared<float>(4, q_ct1);
  alpha[0] = 0;
  alpha[1] = 1;
  alpha[2] = 2;
  alpha[3] = 3;

  // Matmul
  igemmlt<COL_AMPERE, 8, 1>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4, C_col32,
                            alpha, m * 32, ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  q_ct1.wait();

  // Check result
  int8_t Chost[m * n];
  q_ct1.memcpy(Chost, Cdev, m * n * sizeof(int8_t)).wait();

  bool error = false;
  int8_t C_ref[m * n] = {0, 17, 40, 69, 0, 6, 16, 30};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  delete (ltHandle);
  delete (Adesc_col32);
  delete (Bdesc_col32_2r_4r4);
  delete (Cdesc_col32);
  delete (Adesc_col_major);
  delete (Bdesc_col_major);
  delete (Cdesc_col_major);
  dpct::dpct_free(Adev, q_ct1);
  dpct::dpct_free(Bdev, q_ct1);
  dpct::dpct_free(Cdev, q_ct1);
  dpct::dpct_free(alpha, q_ct1);

  return !error;
}

// clang-format off
// A (4*3)    B (2*3)
// 6 10 14    5 -3 1
// 7 11 15    4 -2 0
// 8 12 16
// 9 13 17
//
// alpha * A          * op(B)   = alpha * C       =  C
// 0       6  10  14    5  4      0       14  4      0   0
// 1       7  11  15   -3 -2      1       17  6      17  6
// 2       8  12  16    1  0      2       20  8      40  16
// 3       9  13  17              3       23  10     69  30
//
// alpha * A          * op(B)   = alpha * C       =  C
// 1       6  10  14    5  4      1       14  4      14  4
//         7  11  15   -3 -2              17  6      17  6
//         8  12  16    1  0              20  8      20  8
//         9  13  17                      23  10     23  10
// clang-format on

int main() {
  bool pass = true;
  pass = test1() && pass;
  pass = test2() && pass;
  pass = test3() && pass;
  pass = test4() && pass;
  pass = test5() && pass;
  pass = test6() && pass;
  return pass ? 0 : 1;
}
