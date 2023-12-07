// ===------- sparse_utils_4_usm.cpp ------------------------ *- C++ -* ----===//
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    h_data = (float *)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    d_data =
        (d_data_t *)sycl::malloc_device(sizeof(d_data_t) * element_num, q_ct1);
    q_ct1.memset(d_data, 0, sizeof(d_data_t) * element_num).wait();
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    h_data = (float *)malloc(sizeof(float) * element_num);
    d_data =
        (d_data_t *)sycl::malloc_device(sizeof(d_data_t) * element_num, q_ct1);
    q_ct1.memset(d_data, 0, sizeof(d_data_t) * element_num).wait();
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    sycl::free(d_data, dpct::get_default_queue());
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    dpct::get_default_queue()
        .memcpy(d_data, h_temp, sizeof(d_data_t) * element_num)
        .wait();
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::get_default_queue()
        .memcpy(h_temp, d_data, sizeof(d_data_t) * element_num)
        .wait();
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
void test_cusparseCsrsvEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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

  dpct::sparse::optimize_csrsv(
      *handle, oneapi::mkl::transpose::nontrans, 3, descrA, a_s_val.d_data,
      dpct::library_data_t::real_float, (int *)a_row_ptr_s.d_data,
      (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsv(
      *handle, oneapi::mkl::transpose::nontrans, 3, descrA, a_d_val.d_data,
      dpct::library_data_t::real_double, (int *)a_row_ptr_d.d_data,
      (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsv(
      *handle, oneapi::mkl::transpose::nontrans, 3, descrA, a_c_val.d_data,
      dpct::library_data_t::complex_float, (int *)a_row_ptr_c.d_data,
      (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsv(
      *handle, oneapi::mkl::transpose::nontrans, 3, descrA, a_z_val.d_data,
      dpct::library_data_t::complex_double, (int *)a_row_ptr_z.d_data,
      (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s,
                      dpct::library_data_t::real_float, descrA, a_s_val.d_data,
                      dpct::library_data_t::real_float,
                      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data,
                      info_s, f_s.d_data, dpct::library_data_t::real_float,
                      x_s.d_data, dpct::library_data_t::real_float);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_d,
                      dpct::library_data_t::real_double, descrA, a_d_val.d_data,
                      dpct::library_data_t::real_double,
                      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data,
                      info_d, f_d.d_data, dpct::library_data_t::real_double,
                      x_d.d_data, dpct::library_data_t::real_double);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_c,
                      dpct::library_data_t::complex_float, descrA,
                      a_c_val.d_data, dpct::library_data_t::complex_float,
                      (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data,
                      info_c, f_c.d_data, dpct::library_data_t::complex_float,
                      x_c.d_data, dpct::library_data_t::complex_float);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_z,
                      dpct::library_data_t::complex_double, descrA,
                      a_z_val.d_data, dpct::library_data_t::complex_double,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      info_z, f_z.d_data, dpct::library_data_t::complex_double,
                      x_z.d_data, dpct::library_data_t::complex_double);

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
  DPCT1026:34: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_x[4] = {1, 2, 3};
  if (compare_result(expect_x, x_s.h_data, 3) &&
      compare_result(expect_x, x_d.h_data, 3) &&
      compare_result(expect_x, x_c.h_data, 3) &&
      compare_result(expect_x, x_z.h_data, 3))
    printf("CsrsvEx pass\n");
  else {
    printf("CsrsvEx fail\n");
    test_passed = false;
  }
}

// | 1 1 2 0 |
// | 0 1 3 0 |
// | 0 0 1 5 |
void test_cusparseCsr2csc_00() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 7};
  Data<int> a_s_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_d_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_c_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_z_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2, 3};
  Data<int> a_s_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_d_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_c_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_z_col_ind(a_col_ind_vec.data(), 7);

  Data<float> b_s_val(a_val_vec.data(), 7);
  Data<double> b_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> b_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_s_row_ptr.H2D();
  a_d_row_ptr.H2D();
  a_c_row_ptr.H2D();
  a_z_row_ptr.H2D();
  a_s_col_ind.H2D();
  a_d_col_ind.H2D();
  a_c_col_ind.H2D();
  a_z_col_ind.H2D();


  size_t ws_size_s = 0;
  size_t ws_size_d = 0;
  size_t ws_size_c = 0;
  size_t ws_size_z = 0;
  ws_size_s = 0;
  ws_size_d = 0;
  ws_size_c = 0;
  ws_size_z = 0;

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = (void *)sycl::malloc_device(ws_size_s, q_ct1);
  ws_d = (void *)sycl::malloc_device(ws_size_d, q_ct1);
  ws_c = (void *)sycl::malloc_device(ws_size_c, q_ct1);
  ws_z = (void *)sycl::malloc_device(ws_size_z, q_ct1);

  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data,
                        a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data,
                        b_s_row_ind.d_data, dpct::library_data_t::real_float,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data,
                        a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data,
                        b_d_row_ind.d_data, dpct::library_data_t::real_double,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data,
                        a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data,
                        b_c_row_ind.d_data, dpct::library_data_t::complex_float,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data,
                        a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data,
                        b_z_row_ind.d_data,
                        dpct::library_data_t::complex_double,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);

  b_s_val.D2H();
  b_d_val.D2H();
  b_c_val.D2H();
  b_z_val.D2H();
  b_s_col_ptr.D2H();
  b_d_col_ptr.D2H();
  b_c_col_ptr.D2H();
  b_z_col_ptr.D2H();
  b_s_row_ind.D2H();
  b_d_row_ind.D2H();
  b_c_row_ind.D2H();
  b_z_row_ind.D2H();

  q_ct1.wait();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  handle = nullptr;

  float expect_b_val[7] = {1, 1, 1, 2, 3, 1, 5};
  float expect_b_col_ptr[5] = {0, 1, 3, 6, 7};
  float expect_b_row_ind[7] = {0, 0, 1, 0, 1, 2, 2};
  if (compare_result(expect_b_val, b_s_val.h_data, 7) &&
      compare_result(expect_b_val, b_d_val.h_data, 7) &&
      compare_result(expect_b_val, b_c_val.h_data, 7) &&
      compare_result(expect_b_val, b_z_val.h_data, 7) &&
      compare_result(expect_b_col_ptr, b_s_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_d_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_c_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_z_col_ptr.h_data, 5) &&
      compare_result(expect_b_row_ind, b_s_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_d_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_c_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_z_row_ind.h_data, 7))
    printf("Csr2csc 00 pass\n");
  else {
    printf("Csr2csc 00 fail\n");
    test_passed = false;
  }
}

// | 1 1 2 0 |
// | 0 1 3 0 |
// | 0 0 1 5 |
void test_cusparseCsr2csc_01() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 7};
  Data<int> a_s_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_d_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_c_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_z_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2, 3};
  Data<int> a_s_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_d_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_c_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_z_col_ind(a_col_ind_vec.data(), 7);

  Data<float> b_s_val(a_val_vec.data(), 7);
  Data<double> b_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> b_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_s_row_ptr.H2D();
  a_d_row_ptr.H2D();
  a_c_row_ptr.H2D();
  a_z_row_ptr.H2D();
  a_s_col_ind.H2D();
  a_d_col_ind.H2D();
  a_c_col_ind.H2D();
  a_z_col_ind.H2D();


  size_t ws_size_s = 0;
  size_t ws_size_d = 0;
  size_t ws_size_c = 0;
  size_t ws_size_z = 0;
  ws_size_s = 0;
  ws_size_d = 0;
  ws_size_c = 0;
  ws_size_z = 0;

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = (void *)sycl::malloc_device(ws_size_s, q_ct1);
  ws_d = (void *)sycl::malloc_device(ws_size_d, q_ct1);
  ws_c = (void *)sycl::malloc_device(ws_size_c, q_ct1);
  ws_z = (void *)sycl::malloc_device(ws_size_z, q_ct1);

  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data,
                        a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data,
                        b_s_row_ind.d_data, dpct::library_data_t::real_float,
                        dpct::sparse::conversion_scope::index,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data,
                        a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data,
                        b_d_row_ind.d_data, dpct::library_data_t::real_double,
                        dpct::sparse::conversion_scope::index,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data,
                        a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data,
                        b_c_row_ind.d_data, dpct::library_data_t::complex_float,
                        dpct::sparse::conversion_scope::index,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(
      *handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data,
      b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data,
      dpct::library_data_t::complex_double,
      dpct::sparse::conversion_scope::index, oneapi::mkl::index_base::zero);

  b_s_val.D2H();
  b_d_val.D2H();
  b_c_val.D2H();
  b_z_val.D2H();
  b_s_col_ptr.D2H();
  b_d_col_ptr.D2H();
  b_c_col_ptr.D2H();
  b_z_col_ptr.D2H();
  b_s_row_ind.D2H();
  b_d_row_ind.D2H();
  b_c_row_ind.D2H();
  b_z_row_ind.D2H();

  q_ct1.wait();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  handle = nullptr;

  float expect_b_val[7] = {0, 0, 0, 0, 0, 0, 0};
  float expect_b_col_ptr[5] = {0, 1, 3, 6, 7};
  float expect_b_row_ind[7] = {0, 0, 1, 0, 1, 2, 2};
  if (compare_result(expect_b_val, b_s_val.h_data, 7) &&
      compare_result(expect_b_val, b_d_val.h_data, 7) &&
      compare_result(expect_b_val, b_c_val.h_data, 7) &&
      compare_result(expect_b_val, b_z_val.h_data, 7) &&
      compare_result(expect_b_col_ptr, b_s_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_d_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_c_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_z_col_ptr.h_data, 5) &&
      compare_result(expect_b_row_ind, b_s_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_d_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_c_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_z_row_ind.h_data, 7))
    printf("Csr2csc 01 pass\n");
  else {
    printf("Csr2csc 01 fail\n");
    test_passed = false;
  }
}

// | 1 1 2 0 |
// | 0 1 3 0 |
// | 0 0 1 5 |
void test_cusparseTcsr2csc_00() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 7};
  Data<int> a_s_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_d_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_c_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_z_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2, 3};
  Data<int> a_s_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_d_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_c_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_z_col_ind(a_col_ind_vec.data(), 7);

  Data<float> b_s_val(a_val_vec.data(), 7);
  Data<double> b_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> b_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_s_row_ptr.H2D();
  a_d_row_ptr.H2D();
  a_c_row_ptr.H2D();
  a_z_row_ptr.H2D();
  a_s_col_ind.H2D();
  a_d_col_ind.H2D();
  a_c_col_ind.H2D();
  a_z_col_ind.H2D();

  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data,
                        a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data,
                        b_s_row_ind.d_data,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data,
                        a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data,
                        b_d_row_ind.d_data,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data,
                        a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data,
                        b_c_row_ind.d_data,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(*handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data,
                        a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data,
                        b_z_row_ind.d_data,
                        dpct::sparse::conversion_scope::index_and_value,
                        oneapi::mkl::index_base::zero);

  b_s_val.D2H();
  b_d_val.D2H();
  b_c_val.D2H();
  b_z_val.D2H();
  b_s_col_ptr.D2H();
  b_d_col_ptr.D2H();
  b_c_col_ptr.D2H();
  b_z_col_ptr.D2H();
  b_s_row_ind.D2H();
  b_d_row_ind.D2H();
  b_c_row_ind.D2H();
  b_z_row_ind.D2H();

  q_ct1.wait();
  handle = nullptr;

  float expect_b_val[7] = {1, 1, 1, 2, 3, 1, 5};
  float expect_b_col_ptr[5] = {0, 1, 3, 6, 7};
  float expect_b_row_ind[7] = {0, 0, 1, 0, 1, 2, 2};
  if (compare_result(expect_b_val, b_s_val.h_data, 7) &&
      compare_result(expect_b_val, b_d_val.h_data, 7) &&
      compare_result(expect_b_val, b_c_val.h_data, 7) &&
      compare_result(expect_b_val, b_z_val.h_data, 7) &&
      compare_result(expect_b_col_ptr, b_s_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_d_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_c_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_z_col_ptr.h_data, 5) &&
      compare_result(expect_b_row_ind, b_s_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_d_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_c_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_z_row_ind.h_data, 7))
    printf("Tcsr2csc 00 pass\n");
  else {
    printf("Tcsr2csc 00 fail\n");
    test_passed = false;
  }
}

// | 1 1 2 0 |
// | 0 1 3 0 |
// | 0 0 1 5 |
void test_cusparseTcsr2csc_01() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 7};
  Data<int> a_s_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_d_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_c_row_ptr(a_row_ptr_vec.data(), 4);
  Data<int> a_z_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2, 3};
  Data<int> a_s_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_d_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_c_col_ind(a_col_ind_vec.data(), 7);
  Data<int> a_z_col_ind(a_col_ind_vec.data(), 7);

  Data<float> b_s_val(a_val_vec.data(), 7);
  Data<double> b_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> b_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_s_row_ptr.H2D();
  a_d_row_ptr.H2D();
  a_c_row_ptr.H2D();
  a_z_row_ptr.H2D();
  a_s_col_ind.H2D();
  a_d_col_ind.H2D();
  a_c_col_ind.H2D();
  a_z_col_ind.H2D();

  dpct::sparse::csr2csc(
      *handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data,
      b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data,
      dpct::sparse::conversion_scope::index, oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(
      *handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data,
      b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data,
      dpct::sparse::conversion_scope::index, oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(
      *handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data,
      b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data,
      dpct::sparse::conversion_scope::index, oneapi::mkl::index_base::zero);
  dpct::sparse::csr2csc(
      *handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data,
      b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data,
      dpct::sparse::conversion_scope::index, oneapi::mkl::index_base::zero);

  b_s_val.D2H();
  b_d_val.D2H();
  b_c_val.D2H();
  b_z_val.D2H();
  b_s_col_ptr.D2H();
  b_d_col_ptr.D2H();
  b_c_col_ptr.D2H();
  b_z_col_ptr.D2H();
  b_s_row_ind.D2H();
  b_d_row_ind.D2H();
  b_c_row_ind.D2H();
  b_z_row_ind.D2H();

  q_ct1.wait();
  handle = nullptr;

  float expect_b_val[7] = {0, 0, 0, 0, 0, 0, 0};
  float expect_b_col_ptr[5] = {0, 1, 3, 6, 7};
  float expect_b_row_ind[7] = {0, 0, 1, 0, 1, 2, 2};
  if (compare_result(expect_b_val, b_s_val.h_data, 7) &&
      compare_result(expect_b_val, b_d_val.h_data, 7) &&
      compare_result(expect_b_val, b_c_val.h_data, 7) &&
      compare_result(expect_b_val, b_z_val.h_data, 7) &&
      compare_result(expect_b_col_ptr, b_s_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_d_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_c_col_ptr.h_data, 5) &&
      compare_result(expect_b_col_ptr, b_z_col_ptr.h_data, 5) &&
      compare_result(expect_b_row_ind, b_s_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_d_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_c_row_ind.h_data, 7) &&
      compare_result(expect_b_row_ind, b_z_row_ind.h_data, 7))
    printf("Tcsr2csc 01 pass\n");
  else {
    printf("Tcsr2csc 01 fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseCsrsvEx();
  test_cusparseCsr2csc_00();
  test_cusparseCsr2csc_01();
  test_cusparseTcsr2csc_00();
  test_cusparseTcsr2csc_01();

  if (test_passed)
    return 0;
  return -1;
}
