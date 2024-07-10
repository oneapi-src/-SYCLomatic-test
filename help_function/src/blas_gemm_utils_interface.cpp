// ===------ blas_gemm_utils_interface.cpp ----------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_gemm_utils.hpp>
#include <dpct/lib_common_utils.hpp>

void foo1 () {
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  ltHandle = new dpct::blas_gemm::experimental::descriptor();
  delete (ltHandle);

  dpct::blas_gemm::experimental::matrix_layout_ptr matLayout;
  dpct::library_data_t type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
  matLayout =
      new dpct::blas_gemm::experimental::matrix_layout_t(type, rows, cols, ld);

  dpct::blas_gemm::experimental::matrix_layout_t::attribute attr1;
  void *buf1;
  size_t sizeInBytes1;
  size_t *sizeWritten1;
  matLayout->get_attribute(attr1, buf1);
  matLayout->set_attribute(attr1, buf1);
  delete (matLayout);

  dpct::blas_gemm::experimental::matmul_desc_ptr matmulDesc;
  dpct::compute_type computeType;
  dpct::library_data_t scaleType;
  matmulDesc =
      new dpct::blas_gemm::experimental::matmul_desc_t(computeType, scaleType);

  dpct::blas_gemm::experimental::matmul_desc_t::attribute attr2;
  void *buf2;
  size_t sizeInBytes2;
  size_t *sizeWritten2;
  matmulDesc->get_attribute(attr2, buf2);
  matmulDesc->set_attribute(attr2, buf2);
  delete (matmulDesc);

  int matmulPreference;
  void *buf3;
  size_t sizeInBytes3;
  size_t *sizeWritten3;

  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc;
  dpct::blas_gemm::experimental::matrix_layout_ptr Bdesc;
  dpct::blas_gemm::experimental::matrix_layout_ptr Cdesc;
  dpct::blas_gemm::experimental::matrix_layout_ptr Ddesc;

  int requestedAlgoCount = 1;
  int heuristicResultsArray;
  int returnAlgoCount;
  returnAlgoCount = 1;
}

void foo2() {
  dpct::blas_gemm::experimental::descriptor_ptr lightHandle;
  dpct::blas_gemm::experimental::matmul_desc_ptr computeDesc;
  const void *alpha;
  const void *A;
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc;
  const void *B;
  dpct::blas_gemm::experimental::matrix_layout_ptr Bdesc;
  const void *beta;
  const void *C;
  dpct::blas_gemm::experimental::matrix_layout_ptr Cdesc;
  void *D;
  dpct::blas_gemm::experimental::matrix_layout_ptr Ddesc;
  const int *algo;
  void *workspace;
  size_t workspaceSizeInBytes;
  dpct::queue_ptr stream;
  dpct::blas_gemm::experimental::matmul(lightHandle, computeDesc, alpha, A,
                                        Adesc, B, Bdesc, beta, C, Cdesc, D,
                                        Ddesc, stream);
}

void foo3() {
  dpct::blas_gemm::experimental::order_t a;
  a = dpct::blas_gemm::experimental::order_t::col;
  a = dpct::blas_gemm::experimental::order_t::row;
  a = dpct::blas_gemm::experimental::order_t::col32;
  a = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
  a = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;

  dpct::blas_gemm::experimental::pointer_mode_t b;
  b = dpct::blas_gemm::experimental::pointer_mode_t::host;
  b = dpct::blas_gemm::experimental::pointer_mode_t::device;
  b = dpct::blas_gemm::experimental::pointer_mode_t::device_vector;
  b = dpct::blas_gemm::experimental::pointer_mode_t::
      alpha_device_vector_beta_zero;
  b = dpct::blas_gemm::experimental::pointer_mode_t::
      alpha_device_vector_beta_host;

  dpct::blas_gemm::experimental::matrix_layout_t::attribute c;
  c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::type;
  c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::order;
  c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::rows;
  c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::cols;
  c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::ld;

  dpct::blas_gemm::experimental::matmul_desc_t::attribute d;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::compute_type;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::scale_type;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::pointer_mode;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_a;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_c;
  d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::epilogue;
}

void foo4() {
  dpct::blas_gemm::experimental::transform_desc_ptr transformDesc;
  dpct::library_data_t scaleType;
  transformDesc =
      new dpct::blas_gemm::experimental::transform_desc_t(scaleType);
  oneapi::mkl::transpose opT = oneapi::mkl::transpose::trans;
  size_t sizeWritten;
  transformDesc->set_attribute(
      dpct::blas_gemm::experimental::transform_desc_t::attribute::trans_a,
      &opT);
  transformDesc->get_attribute(
      dpct::blas_gemm::experimental::transform_desc_t::attribute::trans_a,
      &opT);
  delete (transformDesc);

  dpct::blas_gemm::experimental::descriptor_ptr lightHandle;
  const void *alpha;
  const void *A;
  dpct::blas_gemm::experimental::matrix_layout_ptr Adesc;
  const void *beta;
  const void *B;
  dpct::blas_gemm::experimental::matrix_layout_ptr Bdesc;
  void *C;
  dpct::blas_gemm::experimental::matrix_layout_ptr Cdesc;
  dpct::queue_ptr stream;
  dpct::blas_gemm::experimental::matrix_transform(
      transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
}

int main() {
  return 0;
}
