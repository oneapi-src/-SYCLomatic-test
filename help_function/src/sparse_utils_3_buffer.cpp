// ===------- sparse_utils_3_buffer.cpp --------------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/blas_utils.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

template <class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    dpct::dpct_free(d_data);
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    dpct::dpct_memcpy(d_data, h_temp, sizeof(d_data_t) * element_num,
                      dpct::host_to_device);
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::dpct_memcpy(h_temp, d_data, sizeof(d_data_t) * element_num,
                      dpct::device_to_host);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }

private:
  inline void from_float_convert(float *in, d_data_t *out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t *in, float *out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<sycl::float2>::from_float_convert(float *in,
                                                   sycl::float2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}
template <>
inline void Data<sycl::double2>::from_float_convert(float *in,
                                                    sycl::double2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}

template <>
inline void Data<sycl::float2>::to_float_convert(sycl::float2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
}
template <>
inline void Data<sycl::double2>::to_float_convert(sycl::double2 *in,
                                                  float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
}

bool compare_result(float *expect, float *result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i] - expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool compare_result(float *expect, float *result, std::vector<int> indices) {
  for (int i = 0; i < indices.size(); i++) {
    if (std::abs(result[indices[i]] - expect[indices[i]]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 6);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 6);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6};
  Data<int> a_row_ptr_s(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_d(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_c(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_z(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2};
  Data<int> a_col_ind_s(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_d(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_c(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_z(a_col_ind_vec.data(), 6);

  std::vector<float> f_vec = {9, 11, 3};
  Data<float> f_s(f_vec.data(), 3);
  Data<double> f_d(f_vec.data(), 3);
  Data<sycl::float2> f_c(f_vec.data(), 3);
  Data<sycl::double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<sycl::float2> x_c(3);
  Data<sycl::double2> x_z(3);

  sycl::queue *handle;
  handle = &q_ct1;
  std::shared_ptr<dpct::sparse::optimize_info> info_s;
  std::shared_ptr<dpct::sparse::optimize_info> info_d;
  std::shared_ptr<dpct::sparse::optimize_info> info_c;
  std::shared_ptr<dpct::sparse::optimize_info> info_z;
  info_s = std::make_shared<dpct::sparse::optimize_info>();
  info_d = std::make_shared<dpct::sparse::optimize_info>();
  info_c = std::make_shared<dpct::sparse::optimize_info>();
  info_z = std::make_shared<dpct::sparse::optimize_info>();

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_diag(oneapi::mkl::diag::unit);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr_s.H2D();
  a_row_ptr_d.H2D();
  a_row_ptr_c.H2D();
  a_row_ptr_z.H2D();
  a_col_ind_s.H2D();
  a_col_ind_d.H2D();
  a_col_ind_c.H2D();
  a_col_ind_z.H2D();
  f_s.H2D();
  f_d.H2D();
  f_c.H2D();
  f_z.H2D();

  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (float *)a_s_val.d_data,
                               (int *)a_row_ptr_s.d_data,
                               (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (double *)a_d_val.d_data,
                               (int *)a_row_ptr_d.d_data,
                               (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::float2 *)a_c_val.d_data,
                               (int *)a_row_ptr_c.d_data,
                               (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::double2 *)a_z_val.d_data,
                               (int *)a_row_ptr_z.d_data,
                               (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s,
                      descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data,
                      info_s, f_s.d_data, x_s.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_d,
                      descrA, (double *)a_d_val.d_data,
                      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data,
                      info_d, f_d.d_data, x_d.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_c,
                      descrA, (sycl::float2 *)a_c_val.d_data,
                      (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data,
                      info_c, f_c.d_data, x_c.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_z,
                      descrA, (sycl::double2 *)a_z_val.d_data,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      info_z, f_z.d_data, x_z.d_data);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  q_ct1.wait();
  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  /*
  DPCT1026:24: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_x[4] = {1, 2, 3};
  if (compare_result(expect_x, x_s.h_data, 3) &&
      compare_result(expect_x, x_d.h_data, 3) &&
      compare_result(expect_x, x_c.h_data, 3) &&
      compare_result(expect_x, x_z.h_data, 3))
    printf("Tcsrsv pass\n");
  else {
    printf("Tcsrsv fail\n");
    test_passed = false;
  }
}

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 6);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 6);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6};
  Data<int> a_row_ptr_s(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_d(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_c(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_z(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2};
  Data<int> a_col_ind_s(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_d(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_c(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_z(a_col_ind_vec.data(), 6);

  std::vector<float> f_vec = {9, 11, 3};
  Data<float> f_s(f_vec.data(), 3);
  Data<double> f_d(f_vec.data(), 3);
  Data<sycl::float2> f_c(f_vec.data(), 3);
  Data<sycl::double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<sycl::float2> x_c(3);
  Data<sycl::double2> x_z(3);

  sycl::queue *handle;
  handle = &q_ct1;
  std::shared_ptr<dpct::sparse::optimize_info> info_s;
  std::shared_ptr<dpct::sparse::optimize_info> info_d;
  std::shared_ptr<dpct::sparse::optimize_info> info_c;
  std::shared_ptr<dpct::sparse::optimize_info> info_z;
  info_s = std::make_shared<dpct::sparse::optimize_info>();
  info_d = std::make_shared<dpct::sparse::optimize_info>();
  info_c = std::make_shared<dpct::sparse::optimize_info>();
  info_z = std::make_shared<dpct::sparse::optimize_info>();
  int policy = 1;
  policy = 0;

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_diag(oneapi::mkl::diag::unit);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr_s.H2D();
  a_row_ptr_d.H2D();
  a_row_ptr_c.H2D();
  a_row_ptr_z.H2D();
  a_col_ind_s.H2D();
  a_col_ind_d.H2D();
  a_col_ind_c.H2D();
  a_col_ind_z.H2D();
  f_s.H2D();
  f_d.H2D();
  f_c.H2D();
  f_z.H2D();

  int buffer_size_s0;
  int buffer_size_d0;
  int buffer_size_c0;
  int buffer_size_z0;
  buffer_size_s0 = 0;
  buffer_size_d0 = 0;
  buffer_size_c0 = 0;
  buffer_size_z0 = 0;

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  buffer_size_s = 0;
  buffer_size_d = 0;
  buffer_size_c = 0;
  buffer_size_z = 0;

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  buffer_s = dpct::dpct_malloc(buffer_size_s);
  buffer_d = dpct::dpct_malloc(buffer_size_d);
  buffer_c = dpct::dpct_malloc(buffer_size_c);
  buffer_z = dpct::dpct_malloc(buffer_size_z);

  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (float *)a_s_val.d_data,
                               (int *)a_row_ptr_s.d_data,
                               (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (double *)a_d_val.d_data,
                               (int *)a_row_ptr_d.d_data,
                               (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::float2 *)a_c_val.d_data,
                               (int *)a_row_ptr_c.d_data,
                               (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::double2 *)a_z_val.d_data,
                               (int *)a_row_ptr_z.d_data,
                               (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s,
                      descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data,
                      info_s, f_s.d_data, x_s.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_d,
                      descrA, (double *)a_d_val.d_data,
                      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data,
                      info_d, f_d.d_data, x_d.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_c,
                      descrA, (sycl::float2 *)a_c_val.d_data,
                      (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data,
                      info_c, f_c.d_data, x_c.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_z,
                      descrA, (sycl::double2 *)a_z_val.d_data,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      info_z, f_z.d_data, x_z.d_data);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  q_ct1.wait();
  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  /*
  DPCT1026:33: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(buffer_s);
  dpct::dpct_free(buffer_d);
  dpct::dpct_free(buffer_c);
  dpct::dpct_free(buffer_z);

  float expect_x[4] = {1, 2, 3};
  if (compare_result(expect_x, x_s.h_data, 3) &&
      compare_result(expect_x, x_d.h_data, 3) &&
      compare_result(expect_x, x_c.h_data, 3) &&
      compare_result(expect_x, x_z.h_data, 3))
    printf("Tcsrsv2 pass\n");
  else {
    printf("Tcsrsv2 fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrmm2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Data<float> b_s(b_vec.data(), 10);
  Data<double> b_d(b_vec.data(), 10);
  Data<sycl::float2> b_c(b_vec.data(), 10);
  Data<sycl::double2> b_z(b_vec.data(), 10);

  Data<float> c_s(8);
  Data<double> c_d(8);
  Data<sycl::float2> c_c(8);
  Data<sycl::double2> c_z(8);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:32: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

  /*
  DPCT1045:33: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_s.d_data,
                      descrA, a_s_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_s.d_data, 5, beta_s.d_data,
                      c_s.d_data, 4);
  /*
  DPCT1045:34: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_d.d_data,
                      descrA, a_d_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_d.d_data, 5, beta_d.d_data,
                      c_d.d_data, 4);
  /*
  DPCT1045:35: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_c.d_data,
                      descrA, a_c_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_c.d_data, 5, beta_c.d_data,
                      c_c.d_data, 4);
  /*
  DPCT1045:36: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_z.d_data,
                      descrA, a_z_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_z.d_data, 5, beta_z.d_data,
                      c_z.d_data, 4);

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:37: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8) &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8))
    printf("Tcsrmm2 pass\n");
  else {
    printf("Tcsrmm2 fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseTcsrsv();
  test_cusparseTcsrsv2();
  test_cusparseTcsrmm2();

  if (test_passed)
    return 0;
  return -1;
}
