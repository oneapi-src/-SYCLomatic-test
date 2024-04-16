// ===------- sparse_utils_2_usm.cpp ------------------------ *- C++ -* ----===//
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

const bool run_complex_datatype = true;

void test_cusparseSetGetStream() {
  sycl::queue *handle;
  handle = &dpct::get_default_queue();
  dpct::queue_ptr stream;
  stream = handle;
  handle = stream;
  handle = nullptr;
  printf("SetGetStream pass\n");
}

void test_cusparseTcsrmv_ge() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:0: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  /*
  DPCT1045:1: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:2: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:4: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:5: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 5,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:3: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_ge pass\n");
  else {
    printf("Tcsrmv_ge fail\n");
    test_passed = false;
  }
}


//  alpha  *  A          *  B      =  C
//     10  * | 1 4 0 1 |    | 1 |    | 130 |
//           | 4 2 3 0 |    | 2 |    | 170 |
//           | 0 3 0 7 |    | 3 |    | 340 |
//           | 1 0 7 0 |    | 4 |    | 220 |
void test_cusparseTcsrmv_sy() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 1, 4, 2, 3, 3, 7, 1, 7};
  Data<float> a_s_val(a_val_vec.data(), 10);
  Data<double> a_d_val(a_val_vec.data(), 10);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 10);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 10);
  std::vector<float> a_row_ptr_vec = {0, 3, 6, 8, 10};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 0, 1, 2, 1, 3, 0, 2};
  Data<int> a_col_ind(a_col_ind_vec.data(), 10);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<sycl::float2> b_c(b_vec.data(), 4);
  Data<sycl::double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:6: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::sy);

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

  /*
  DPCT1045:7: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:8: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:10: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:11: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 4,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:9: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {130, 170, 340, 220};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_sy pass\n");
  else {
    printf("Tcsrmv_sy fail\n");
    test_passed = false;
  }
}

//  alpha  *  A          *  B      =  C
//     10  * | 1 4 0 1 |    | 1 |    | 130 |
//           | 0 2 3 0 |    | 2 |    | 130 |
//           | 0 0 0 7 |    | 3 |    | 280 |
//           | 0 0 0 1 |    | 4 |    | 40  |

// Note: this matrix type is not supported in CUDA but supported in oneMKL
void test_cusparseTcsrmv_tr() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 1, 2, 3, 7, 1};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6, 7};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 1, 2, 3, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 7);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<sycl::float2> b_c(b_vec.data(), 4);
  Data<sycl::double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:12: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_uplo(oneapi::mkl::uplo::upper);

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

  /*
  DPCT1045:13: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:14: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:16: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:17: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 4,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:15: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {130, 130, 280, 40};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_tr pass\n");
  else {
    printf("Tcsrmv_tr fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrmm() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  DPCT1026:18: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  /*
  DPCT1045:19: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, 5, (float *)beta_s.d_data,
                      (float *)c_s.d_data, 4);
  /*
  DPCT1045:20: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data, 5,
                      (double *)beta_d.d_data, (double *)c_d.d_data, 4);
  if (run_complex_datatype) {
    /*
    DPCT1045:22: Migration is only supported for this API for the general sparse
    matrix type. You may need to adjust the code.
    */
    dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data, 5,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data, 4);
    /*
    DPCT1045:23: Migration is only supported for this API for the general sparse
    matrix type. You may need to adjust the code.
    */
    dpct::sparse::csrmm(
        *handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data, 5,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data, 4);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:21: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8) &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8))
    printf("Tcsrmm pass\n");
  else {
    printf("Tcsrmm fail\n");
    test_passed = false;
  }
}

void test_cusparseSpMV() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:0: The call to cusparseSetPointerMode was removed because this call
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

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_s.d_data, dpct::library_data_t::real_float);
  b_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_d.d_data, dpct::library_data_t::real_double);
  b_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_c.d_data, dpct::library_data_t::complex_float);
  b_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_z.d_data, dpct::library_data_t::complex_double);

  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_s.d_data, dpct::library_data_t::real_float);
  c_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_d.d_data, dpct::library_data_t::real_double);
  c_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_c.d_data, dpct::library_data_t::complex_float);
  c_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_z.d_data, dpct::library_data_t::complex_double);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
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

  dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                     a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                     dpct::library_data_t::real_float);
  dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                     a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                     dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data,
                       c_descr_c, dpct::library_data_t::complex_float);
    dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data,
                       c_descr_z, dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("SpMV pass\n");
  else {
    printf("SpMV fail\n");
    test_passed = false;
  }
}

void test_cusparseSpMM() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  DPCT1026:1: The call to cusparseSetPointerMode was removed because this call
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

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_s;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_d;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_c;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_s.d_data, dpct::library_data_t::real_float,
      oneapi::mkl::layout::col_major);
  b_descr_d = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_d.d_data, dpct::library_data_t::real_double,
      oneapi::mkl::layout::col_major);
  b_descr_c = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_c.d_data, dpct::library_data_t::complex_float,
      oneapi::mkl::layout::col_major);
  b_descr_z = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_z.d_data, dpct::library_data_t::complex_double,
      oneapi::mkl::layout::col_major);

  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_s;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_d;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_c;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_s.d_data, dpct::library_data_t::real_float,
      oneapi::mkl::layout::col_major);
  c_descr_d = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_d.d_data, dpct::library_data_t::real_double,
      oneapi::mkl::layout::col_major);
  c_descr_c = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_c.d_data, dpct::library_data_t::complex_float,
      oneapi::mkl::layout::col_major);
  c_descr_z = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_z.d_data, dpct::library_data_t::complex_double,
      oneapi::mkl::layout::col_major);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
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

  /*
  DPCT1026:2: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:3: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:4: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:5: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                     a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                     dpct::library_data_t::real_float);
  dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                     a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                     dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                       oneapi::mkl::transpose::nontrans, alpha_c.d_data,
                       a_descr_c, b_descr_c, beta_c.d_data, c_descr_c,
                       dpct::library_data_t::complex_float);
    dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                       oneapi::mkl::transpose::nontrans, alpha_z.d_data,
                       a_descr_z, b_descr_z, beta_z.d_data, c_descr_z,
                       dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8) &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8))
    printf("SpMM pass\n");
  else {
    printf("SpMM fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrmv_mp() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:25: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  /*
  DPCT1045:26: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:27: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:29: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:30: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 5,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:28: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_mp pass\n");
  else {
    printf("Tcsrmv_mp fail\n");
    test_passed = false;
  }
}

void test_cusparseCsrmvEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

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
  DPCT1026:31: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  int alg;

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  /*
  DPCT1026:32: The call to cusparseCsrmvEx_bufferSize was removed because this
  call is redundant in SYCL.
  */
  /*
  DPCT1026:33: The call to cusparseCsrmvEx_bufferSize was removed because this
  call is redundant in SYCL.
  */
  if (run_complex_datatype) {
    /*
    DPCT1026:35: The call to cusparseCsrmvEx_bufferSize was removed because this
    call is redundant in SYCL.
    */
    /*
    DPCT1026:36: The call to cusparseCsrmvEx_bufferSize was removed because this
    call is redundant in SYCL.
    */
  }

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = (void *)sycl::malloc_device(ws_size_s, q_ct1);
  ws_d = (void *)sycl::malloc_device(ws_size_d, q_ct1);
  ws_c = (void *)sycl::malloc_device(ws_size_c, q_ct1);
  ws_z = (void *)sycl::malloc_device(ws_size_z, q_ct1);

  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      alpha_s.d_data, dpct::library_data_t::real_float, descrA,
                      a_s_val.d_data, dpct::library_data_t::real_float,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      b_s.d_data, dpct::library_data_t::real_float,
                      beta_s.d_data, dpct::library_data_t::real_float,
                      c_s.d_data, dpct::library_data_t::real_float);
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      alpha_d.d_data, dpct::library_data_t::real_double, descrA,
                      a_d_val.d_data, dpct::library_data_t::real_double,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      b_d.d_data, dpct::library_data_t::real_double,
                      beta_d.d_data, dpct::library_data_t::real_double,
                      c_d.d_data, dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                        alpha_c.d_data, dpct::library_data_t::complex_float,
                        descrA, a_c_val.d_data,
                        dpct::library_data_t::complex_float,
                        (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                        b_c.d_data, dpct::library_data_t::complex_float,
                        beta_c.d_data, dpct::library_data_t::complex_float,
                        c_c.d_data, dpct::library_data_t::complex_float);
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                        alpha_z.d_data, dpct::library_data_t::complex_double,
                        descrA, a_z_val.d_data,
                        dpct::library_data_t::complex_double,
                        (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                        b_z.d_data, dpct::library_data_t::complex_double,
                        beta_z.d_data, dpct::library_data_t::complex_double,
                        c_z.d_data, dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  q_ct1.wait();
  /*
  DPCT1026:34: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("CsrmvEx pass\n");
  else {
    printf("CsrmvEx fail\n");
    test_passed = false;
  }
}

// A * B = C
//
// | 0 1 2 |   | 1 0 0 0 |   | 2 3 10 12 |  
// | 0 0 3 | * | 2 3 0 0 | = | 0 0 15 18 |
// | 4 0 0 |   | 0 0 5 6 |   | 4 0 0  0  |
void test_cusparseSpGEMM() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 2, 3, 4};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 4);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 4);
  std::vector<float> a_row_ptr_vec = {0, 2, 3, 4};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {1, 2, 2, 0};
  Data<int> a_col_ind(a_col_ind_vec.data(), 4);

  std::vector<float> b_val_vec = {1, 2, 3, 5, 6};
  Data<float> b_s_val(b_val_vec.data(), 5);
  Data<double> b_d_val(b_val_vec.data(), 5);
  Data<sycl::float2> b_c_val(b_val_vec.data(), 5);
  Data<sycl::double2> b_z_val(b_val_vec.data(), 5);
  std::vector<float> b_row_ptr_vec = {0, 1, 3, 5};
  Data<int> b_row_ptr(b_row_ptr_vec.data(), 4);
  std::vector<float> b_col_ind_vec = {0, 0, 1, 2, 3};
  Data<int> b_col_ind(b_col_ind_vec.data(), 5);

  float alpha = 1;
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
  DPCT1026:0: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s_val.H2D();
  b_d_val.H2D();
  b_c_val.H2D();
  b_z_val.H2D();
  b_row_ptr.H2D();
  b_col_ind.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  dpct::sparse::sparse_matrix_desc_t b_descr_s;
  dpct::sparse::sparse_matrix_desc_t b_descr_d;
  dpct::sparse::sparse_matrix_desc_t b_descr_c;
  dpct::sparse::sparse_matrix_desc_t b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  b_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  b_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  b_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  Data<int> c_s_row_ptr(4);
  Data<int> c_d_row_ptr(4);
  Data<int> c_c_row_ptr(4);
  Data<int> c_z_row_ptr(4);

  dpct::sparse::sparse_matrix_desc_t c_descr_s;
  dpct::sparse::sparse_matrix_desc_t c_descr_d;
  dpct::sparse::sparse_matrix_desc_t c_descr_c;
  dpct::sparse::sparse_matrix_desc_t c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 0, c_s_row_ptr.d_data, nullptr, nullptr,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  c_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 0, c_d_row_ptr.d_data, nullptr, nullptr,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  c_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 0, c_c_row_ptr.d_data, nullptr, nullptr,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  c_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 4, 0, c_z_row_ptr.d_data, nullptr, nullptr,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  oneapi::mkl::sparse::matmat_descr_t SpGEMMDescr_s;
  oneapi::mkl::sparse::matmat_descr_t SpGEMMDescr_d;
  oneapi::mkl::sparse::matmat_descr_t SpGEMMDescr_c;
  oneapi::mkl::sparse::matmat_descr_t SpGEMMDescr_z;
  oneapi::mkl::sparse::init_matmat_descr(&SpGEMMDescr_s);
  oneapi::mkl::sparse::init_matmat_descr(&SpGEMMDescr_d);
  oneapi::mkl::sparse::init_matmat_descr(&SpGEMMDescr_c);
  oneapi::mkl::sparse::init_matmat_descr(&SpGEMMDescr_z);

  size_t ws_1_size_s = 0;
  size_t ws_1_size_d = 0;
  size_t ws_1_size_c = 0;
  size_t ws_1_size_z = 0;
  dpct::sparse::spgemm_work_estimation(
      *handle, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, alpha_s.d_data, a_descr_s, b_descr_s,
      beta_s.d_data, c_descr_s, SpGEMMDescr_s, &ws_1_size_s, NULL);
  dpct::sparse::spgemm_work_estimation(
      *handle, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, alpha_d.d_data, a_descr_d, b_descr_d,
      beta_d.d_data, c_descr_d, SpGEMMDescr_d, &ws_1_size_d, NULL);
  if (run_complex_datatype) {
    dpct::sparse::spgemm_work_estimation(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_c.d_data, a_descr_c, b_descr_c,
        beta_c.d_data, c_descr_c, SpGEMMDescr_c, &ws_1_size_c, NULL);
    dpct::sparse::spgemm_work_estimation(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_z.d_data, a_descr_z, b_descr_z,
        beta_z.d_data, c_descr_z, SpGEMMDescr_z, &ws_1_size_z, NULL);
  }

  void *ws_1_s;
  void *ws_1_d;
  void *ws_1_c;
  void *ws_1_z;
  ws_1_s = (void *)sycl::malloc_device(ws_1_size_s, q_ct1);
  ws_1_d = (void *)sycl::malloc_device(ws_1_size_d, q_ct1);
  ws_1_c = (void *)sycl::malloc_device(ws_1_size_c, q_ct1);
  ws_1_z = (void *)sycl::malloc_device(ws_1_size_z, q_ct1);

  dpct::sparse::spgemm_work_estimation(
      *handle, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, alpha_s.d_data, a_descr_s, b_descr_s,
      beta_s.d_data, c_descr_s, SpGEMMDescr_s, &ws_1_size_s, ws_1_s);
  dpct::sparse::spgemm_work_estimation(
      *handle, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, alpha_d.d_data, a_descr_d, b_descr_d,
      beta_d.d_data, c_descr_d, SpGEMMDescr_d, &ws_1_size_d, ws_1_d);
  if (run_complex_datatype) {
    dpct::sparse::spgemm_work_estimation(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_c.d_data, a_descr_c, b_descr_c,
        beta_c.d_data, c_descr_c, SpGEMMDescr_c, &ws_1_size_c, ws_1_c);
    dpct::sparse::spgemm_work_estimation(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_z.d_data, a_descr_z, b_descr_z,
        beta_z.d_data, c_descr_z, SpGEMMDescr_z, &ws_1_size_z, ws_1_z);
  }

  size_t ws_2_size_s = 0;
  size_t ws_2_size_d = 0;
  size_t ws_2_size_c = 0;
  size_t ws_2_size_z = 0;
  dpct::sparse::spgemm_compute(*handle, oneapi::mkl::transpose::nontrans,
                               oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                               a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                               SpGEMMDescr_s, &ws_2_size_s, NULL);
  dpct::sparse::spgemm_compute(*handle, oneapi::mkl::transpose::nontrans,
                               oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                               a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                               SpGEMMDescr_d, &ws_2_size_d, NULL);
  if (run_complex_datatype) {
    dpct::sparse::spgemm_compute(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_c.d_data, a_descr_c, b_descr_c,
        beta_c.d_data, c_descr_c, SpGEMMDescr_c, &ws_2_size_c, NULL);
    dpct::sparse::spgemm_compute(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_z.d_data, a_descr_z, b_descr_z,
        beta_z.d_data, c_descr_z, SpGEMMDescr_z, &ws_2_size_z, NULL);
  }

  void *ws_2_s;
  void *ws_2_d;
  void *ws_2_c;
  void *ws_2_z;
  ws_2_s = (void *)sycl::malloc_device(ws_2_size_s, q_ct1);
  ws_2_d = (void *)sycl::malloc_device(ws_2_size_d, q_ct1);
  ws_2_c = (void *)sycl::malloc_device(ws_2_size_c, q_ct1);
  ws_2_z = (void *)sycl::malloc_device(ws_2_size_z, q_ct1);

  dpct::sparse::spgemm_compute(*handle, oneapi::mkl::transpose::nontrans,
                               oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                               a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                               SpGEMMDescr_s, &ws_2_size_s, ws_2_s);
  dpct::sparse::spgemm_compute(*handle, oneapi::mkl::transpose::nontrans,
                               oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                               a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                               SpGEMMDescr_d, &ws_2_size_d, ws_2_d);
  if (run_complex_datatype) {
    dpct::sparse::spgemm_compute(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_c.d_data, a_descr_c, b_descr_c,
        beta_c.d_data, c_descr_c, SpGEMMDescr_c, &ws_2_size_c, ws_2_c);
    dpct::sparse::spgemm_compute(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, alpha_z.d_data, a_descr_z, b_descr_z,
        beta_z.d_data, c_descr_z, SpGEMMDescr_z, &ws_2_size_z, ws_2_z);
  }

  int64_t c_row_s;
  int64_t c_row_d;
  int64_t c_row_c;
  int64_t c_row_z;
  int64_t c_col_s;
  int64_t c_col_d;
  int64_t c_col_c;
  int64_t c_col_z;
  int64_t c_nnz_s;
  int64_t c_nnz_d;
  int64_t c_nnz_c;
  int64_t c_nnz_z;
  c_descr_s->get_size(&c_row_s, &c_col_s, &c_nnz_s);
  c_descr_d->get_size(&c_row_d, &c_col_d, &c_nnz_d);
  c_descr_c->get_size(&c_row_c, &c_col_c, &c_nnz_c);
  c_descr_z->get_size(&c_row_z, &c_col_z, &c_nnz_z);

  Data<float> c_s_val(c_nnz_s);
  Data<double> c_d_val(c_nnz_d);
  Data<sycl::float2> c_c_val(c_nnz_c);
  Data<sycl::double2> c_z_val(c_nnz_z);
  Data<int> c_s_col_ind(c_nnz_s);
  Data<int> c_d_col_ind(c_nnz_d);
  Data<int> c_c_col_ind(c_nnz_c);
  Data<int> c_z_col_ind(c_nnz_z);

  c_descr_s->set_pointers(c_s_row_ptr.d_data, c_s_col_ind.d_data,
                          c_s_val.d_data);
  c_descr_d->set_pointers(c_d_row_ptr.d_data, c_d_col_ind.d_data,
                          c_d_val.d_data);
  c_descr_c->set_pointers(c_c_row_ptr.d_data, c_c_col_ind.d_data,
                          c_c_val.d_data);
  c_descr_z->set_pointers(c_z_row_ptr.d_data, c_z_col_ind.d_data,
                          c_z_val.d_data);

  dpct::sparse::spgemm_finalize(*handle, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                alpha_s.d_data, a_descr_s, b_descr_s,
                                beta_s.d_data, c_descr_s, SpGEMMDescr_s);
  dpct::sparse::spgemm_finalize(*handle, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                alpha_d.d_data, a_descr_d, b_descr_d,
                                beta_d.d_data, c_descr_d, SpGEMMDescr_d);
  if (run_complex_datatype) {
    dpct::sparse::spgemm_finalize(*handle, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans,
                                  alpha_c.d_data, a_descr_c, b_descr_c,
                                  beta_c.d_data, c_descr_c, SpGEMMDescr_c);
    dpct::sparse::spgemm_finalize(*handle, oneapi::mkl::transpose::nontrans,
                                  oneapi::mkl::transpose::nontrans,
                                  alpha_z.d_data, a_descr_z, b_descr_z,
                                  beta_z.d_data, c_descr_z, SpGEMMDescr_z);
  }

  q_ct1.wait();

  sycl::free(ws_1_s, q_ct1);
  sycl::free(ws_1_d, q_ct1);
  sycl::free(ws_1_c, q_ct1);
  sycl::free(ws_1_z, q_ct1);
  sycl::free(ws_2_s, q_ct1);
  sycl::free(ws_2_d, q_ct1);
  sycl::free(ws_2_c, q_ct1);
  sycl::free(ws_2_z, q_ct1);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  oneapi::mkl::sparse::release_matmat_descr(&SpGEMMDescr_s);
  oneapi::mkl::sparse::release_matmat_descr(&SpGEMMDescr_d);
  oneapi::mkl::sparse::release_matmat_descr(&SpGEMMDescr_c);
  oneapi::mkl::sparse::release_matmat_descr(&SpGEMMDescr_z);
  handle = nullptr;

  c_s_val.D2H();
  c_d_val.D2H();
  c_c_val.D2H();
  c_z_val.D2H();
  c_s_row_ptr.D2H();
  c_d_row_ptr.D2H();
  c_c_row_ptr.D2H();
  c_z_row_ptr.D2H();
  c_s_col_ind.D2H();
  c_d_col_ind.D2H();
  c_c_col_ind.D2H();
  c_z_col_ind.D2H();

  float expect_c_val[7] = {2.000000, 3.000000, 10.000000, 12.000000, 15.000000, 18.000000, 4.000000};
  float expect_c_row_ptr[4] = {0.000000, 4.000000, 6.000000, 7.000000};
  float expect_c_col_ind[7] = {0.000000, 1.000000, 2.000000, 3.000000, 2.000000, 3.000000, 0.000000};
  if (compare_result(expect_c_val, c_s_val.h_data, 7) &&
      compare_result(expect_c_val, c_d_val.h_data, 7) &&
      compare_result(expect_c_val, c_c_val.h_data, 7) &&
      compare_result(expect_c_val, c_z_val.h_data, 7) &&
      compare_result(expect_c_row_ptr, c_s_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_d_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_c_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_z_row_ptr.h_data, 4) &&
      compare_result(expect_c_col_ind, c_s_col_ind.h_data, 7) &&
      compare_result(expect_c_col_ind, c_d_col_ind.h_data, 7) &&
      compare_result(expect_c_col_ind, c_c_col_ind.h_data, 7) &&
      compare_result(expect_c_col_ind, c_z_col_ind.h_data, 7)
    )
    printf("SpGEMM pass\n");
  else {
    printf("SpGEMM fail\n");
    test_passed = false;
  }
}

// A * C = B
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseSpSV() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 6);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 6);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2};
  Data<int> a_col_ind(a_col_ind_vec.data(), 6);

  std::vector<float> b_vec = {9, 11, 3};
  Data<float> b_s(b_vec.data(), 3);
  Data<double> b_d(b_vec.data(), 3);
  Data<sycl::float2> b_c(b_vec.data(), 3);
  Data<sycl::double2> b_z(b_vec.data(), 3);

  Data<float> c_s(3);
  Data<double> c_d(3);
  Data<sycl::float2> c_c(3);
  Data<sycl::double2> c_z(3);

  float alpha = 1;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:1: The call to cusparseSetPointerMode was removed because this call
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

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, b_s.d_data, dpct::library_data_t::real_float);
  b_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, b_d.d_data, dpct::library_data_t::real_double);
  b_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, b_c.d_data, dpct::library_data_t::complex_float);
  b_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, b_z.d_data, dpct::library_data_t::complex_double);

  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, c_s.d_data, dpct::library_data_t::real_float);
  c_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, c_d.d_data, dpct::library_data_t::real_double);
  c_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, c_c.d_data, dpct::library_data_t::complex_float);
  c_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      3, c_z.d_data, dpct::library_data_t::complex_double);

  oneapi::mkl::uplo uplo = oneapi::mkl::uplo::upper;
  a_descr_s->set_attribute(dpct::sparse::matrix_attribute::uplo, &uplo,
                           sizeof(uplo));
  a_descr_d->set_attribute(dpct::sparse::matrix_attribute::uplo, &uplo,
                           sizeof(uplo));
  a_descr_c->set_attribute(dpct::sparse::matrix_attribute::uplo, &uplo,
                           sizeof(uplo));
  a_descr_z->set_attribute(dpct::sparse::matrix_attribute::uplo, &uplo,
                           sizeof(uplo));
  oneapi::mkl::diag diag = oneapi::mkl::diag::unit;
  a_descr_s->set_attribute(dpct::sparse::matrix_attribute::diag, &diag,
                           sizeof(diag));
  a_descr_d->set_attribute(dpct::sparse::matrix_attribute::diag, &diag,
                           sizeof(diag));
  a_descr_c->set_attribute(dpct::sparse::matrix_attribute::diag, &diag,
                           sizeof(diag));
  a_descr_z->set_attribute(dpct::sparse::matrix_attribute::diag, &diag,
                           sizeof(diag));

  int SpSVDescr_s;
  int SpSVDescr_d;
  int SpSVDescr_c;
  int SpSVDescr_z;
  /*
  DPCT1026:2: The call to cusparseSpSV_createDescr was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:3: The call to cusparseSpSV_createDescr was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:4: The call to cusparseSpSV_createDescr was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:5: The call to cusparseSpSV_createDescr was removed because this call
  is redundant in SYCL.
  */

  size_t ws_size_s = 0;
  size_t ws_size_d = 0;
  size_t ws_size_c = 0;
  size_t ws_size_z = 0;
  /*
  DPCT1026:6: The call to cusparseSpSV_bufferSize was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:7: The call to cusparseSpSV_bufferSize was removed because this call
  is redundant in SYCL.
  */
  if (run_complex_datatype) {
    /*
    DPCT1026:12: The call to cusparseSpSV_bufferSize was removed because this
    call is redundant in SYCL.
    */
    /*
    DPCT1026:13: The call to cusparseSpSV_bufferSize was removed because this
    call is redundant in SYCL.
    */
  }

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = (void *)sycl::malloc_device(ws_size_s, q_ct1);
  ws_d = (void *)sycl::malloc_device(ws_size_d, q_ct1);
  ws_c = (void *)sycl::malloc_device(ws_size_c, q_ct1);
  ws_z = (void *)sycl::malloc_device(ws_size_z, q_ct1);

  dpct::sparse::spsv_optimize(*handle, oneapi::mkl::transpose::nontrans,
                              a_descr_s);
  dpct::sparse::spsv_optimize(*handle, oneapi::mkl::transpose::nontrans,
                              a_descr_d);
  if (run_complex_datatype) {
    dpct::sparse::spsv_optimize(*handle, oneapi::mkl::transpose::nontrans,
                                a_descr_c);
    dpct::sparse::spsv_optimize(*handle, oneapi::mkl::transpose::nontrans,
                                a_descr_z);
  }

  dpct::sparse::spsv(*handle, oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                     a_descr_s, b_descr_s, c_descr_s,
                     dpct::library_data_t::real_float);
  dpct::sparse::spsv(*handle, oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                     a_descr_d, b_descr_d, c_descr_d,
                     dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::spsv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_c.d_data, a_descr_c, b_descr_c, c_descr_c,
                       dpct::library_data_t::complex_float);
    dpct::sparse::spsv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_z.d_data, a_descr_z, b_descr_z, c_descr_z,
                       dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  sycl::free(ws_s, q_ct1);
  sycl::free(ws_d, q_ct1);
  sycl::free(ws_c, q_ct1);
  sycl::free(ws_z, q_ct1);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  /*
  DPCT1026:8: The call to cusparseSpSV_destroyDescr was removed because this
  call is redundant in SYCL.
  */
  /*
  DPCT1026:9: The call to cusparseSpSV_destroyDescr was removed because this
  call is redundant in SYCL.
  */
  /*
  DPCT1026:10: The call to cusparseSpSV_destroyDescr was removed because this
  call is redundant in SYCL.
  */
  /*
  DPCT1026:11: The call to cusparseSpSV_destroyDescr was removed because this
  call is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {1, 2, 3};
  if (compare_result(expect_c, c_s.h_data, 3) &&
      compare_result(expect_c, c_d.h_data, 3) &&
      compare_result(expect_c, c_c.h_data, 3) &&
      compare_result(expect_c, c_z.h_data, 3))
    printf("SpSV pass\n");
  else {
    printf("SpSV fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseSetGetStream();
  test_cusparseTcsrmv_ge();
  test_cusparseTcsrmv_sy();
  test_cusparseTcsrmv_tr();
  test_cusparseTcsrmm();
  test_cusparseSpMV();
  test_cusparseSpMM();
  test_cusparseTcsrmv_mp();
  test_cusparseCsrmvEx();
  test_cusparseSpGEMM();
  test_cusparseSpSV();

  if (test_passed)
    return 0;
  return -1;
}
