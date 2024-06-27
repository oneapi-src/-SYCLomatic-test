// ===------- sparse_utils_1.cpp ---------------------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/blas_utils.hpp>
#include <cstdio>

bool test1() {
  std::shared_ptr<dpct::sparse::matrix_info> descr;
  dpct::sparse::matrix_info::matrix_type mt;
  oneapi::mkl::diag dt;
  oneapi::mkl::uplo fm;
  oneapi::mkl::index_base ib;

  descr = std::make_shared<dpct::sparse::matrix_info>();

  descr->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  mt = descr->get_matrix_type();
  if (mt != dpct::sparse::matrix_info::matrix_type::ge)
    return false;

  descr->set_matrix_type(dpct::sparse::matrix_info::matrix_type::sy);
  mt = descr->get_matrix_type();
  if (mt != dpct::sparse::matrix_info::matrix_type::sy)
    return false;

  descr->set_matrix_type(dpct::sparse::matrix_info::matrix_type::he);
  mt = descr->get_matrix_type();
  if (mt != dpct::sparse::matrix_info::matrix_type::he)
    return false;

  descr->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  mt = descr->get_matrix_type();
  if (mt != dpct::sparse::matrix_info::matrix_type::tr)
    return false;

  descr->set_diag(oneapi::mkl::diag::nonunit);
  dt = descr->get_diag();
  if (dt != oneapi::mkl::diag::nonunit)
    return false;

  descr->set_diag(oneapi::mkl::diag::unit);
  dt = descr->get_diag();
  if (dt != oneapi::mkl::diag::unit)
    return false;

  descr->set_uplo(oneapi::mkl::uplo::lower);
  fm = descr->get_uplo();
  if (fm != oneapi::mkl::uplo::lower)
    return false;

  descr->set_uplo(oneapi::mkl::uplo::upper);
  fm = descr->get_uplo();
  if (fm != oneapi::mkl::uplo::upper)
    return false;

  descr->set_index_base(oneapi::mkl::index_base::zero);
  ib = descr->get_index_base();
  if (ib != oneapi::mkl::index_base::zero)
    return false;

  descr->set_index_base(oneapi::mkl::index_base::one);
  ib = descr->get_index_base();
  if (ib != oneapi::mkl::index_base::one)
    return false;

  dpct::sparse::matrix_format mf;
  mf = dpct::sparse::matrix_format::csr;
  mf = dpct::sparse::matrix_format::coo;

  return true;
}

int main() {
  bool res = true;

  if ((res = test1())) {
    printf("test1 passed\n");
  } else {
    printf("test1 failed\n");
  }

  return 0;
}
