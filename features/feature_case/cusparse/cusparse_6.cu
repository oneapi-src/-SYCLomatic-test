// ===------- cusparse_6.cu -------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusparse.h"

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
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num,
               cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num,
               cudaMemcpyDeviceToHost);
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
inline void Data<float2>::from_float_convert(float *in, float2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float *in, double2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}

template <>
inline void Data<float2>::to_float_convert(float2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
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
void test_cusparseTcsrsv2() {
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<float2> a_c_val(a_val_vec.data(), 6);
  Data<double2> a_z_val(a_val_vec.data(), 6);
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
  Data<float2> f_c(f_vec.data(), 3);
  Data<double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<float2> x_c(3);
  Data<double2> x_z(3);

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  csrsv2Info_t info_s;
  csrsv2Info_t info_d;
  csrsv2Info_t info_c;
  csrsv2Info_t info_z;
  cusparseCreateCsrsv2Info(&info_s);
  cusparseCreateCsrsv2Info(&info_d);
  cusparseCreateCsrsv2Info(&info_c);
  cusparseCreateCsrsv2Info(&info_z);
  cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;


  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT);

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
  cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, &buffer_size_s0);
  cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, &buffer_size_d0);
  cusparseCcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, &buffer_size_c0);
  cusparseZcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, &buffer_size_z0);

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  cusparseScsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, &buffer_size_s);
  cusparseDcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, &buffer_size_d);
  cusparseCcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, &buffer_size_c);
  cusparseZcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, &buffer_size_z);

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  cudaMalloc(&buffer_s, buffer_size_s);
  cudaMalloc(&buffer_d, buffer_size_d);
  cudaMalloc(&buffer_c, buffer_size_c);
  cudaMalloc(&buffer_z, buffer_size_z);

  cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, policy, buffer_s);
  cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, policy, buffer_d);
  cusparseCcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, policy, buffer_c);
  cusparseZcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, policy, buffer_z);

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_s, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, f_s.d_data, x_s.d_data, policy, buffer_s);
  cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_d, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, f_d.d_data, x_d.d_data, policy, buffer_d);
  cusparseCcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_c, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, f_c.d_data, x_c.d_data, policy, buffer_c);
  cusparseZcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, f_z.d_data, x_z.d_data, policy, buffer_z);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyCsrsv2Info(info_s);
  cusparseDestroyCsrsv2Info(info_d);
  cusparseDestroyCsrsv2Info(info_c);
  cusparseDestroyCsrsv2Info(info_z);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
  cudaFree(buffer_s);
  cudaFree(buffer_d);
  cudaFree(buffer_c);
  cudaFree(buffer_z);

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

int main() {
  test_cusparseTcsrsv2();

  if (test_passed)
    return 0;
  return -1;
}
