// ===------- cusparse_7.cu -------------------------------- *- CUDA -* ----===//
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
void test_cusparseTcsrsv() {
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
  cusparseSolveAnalysisInfo_t info_s;
  cusparseSolveAnalysisInfo_t info_d;
  cusparseSolveAnalysisInfo_t info_c;
  cusparseSolveAnalysisInfo_t info_z;
  cusparseCreateSolveAnalysisInfo(&info_s);
  cusparseCreateSolveAnalysisInfo(&info_d);
  cusparseCreateSolveAnalysisInfo(&info_c);
  cusparseCreateSolveAnalysisInfo(&info_z);

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

  cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s);
  cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d);
  cusparseCcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c);
  cusparseZcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_s, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, f_s.d_data, x_s.d_data);
  cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_d, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, f_d.d_data, x_d.d_data);
  cusparseCcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_c, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, f_c.d_data, x_c.d_data);
  cusparseZcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, f_z.d_data, x_z.d_data);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroySolveAnalysisInfo(info_s);
  cusparseDestroySolveAnalysisInfo(info_d);
  cusparseDestroySolveAnalysisInfo(info_c);
  cusparseDestroySolveAnalysisInfo(info_z);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

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
void test_cusparseCsrsvEx() {
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
  cusparseSolveAnalysisInfo_t info_s;
  cusparseSolveAnalysisInfo_t info_d;
  cusparseSolveAnalysisInfo_t info_c;
  cusparseSolveAnalysisInfo_t info_z;
  cusparseCreateSolveAnalysisInfo(&info_s);
  cusparseCreateSolveAnalysisInfo(&info_d);
  cusparseCreateSolveAnalysisInfo(&info_c);
  cusparseCreateSolveAnalysisInfo(&info_z);

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

  cusparseCsrsv_analysisEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, a_s_val.d_data, CUDA_R_32F, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, CUDA_R_32F);
  cusparseCsrsv_analysisEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, a_d_val.d_data, CUDA_R_64F, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, CUDA_R_64F);
  cusparseCsrsv_analysisEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, a_c_val.d_data, CUDA_C_32F, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, CUDA_C_32F);
  cusparseCsrsv_analysisEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, a_z_val.d_data, CUDA_C_64F, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, CUDA_C_64F);

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseCsrsv_solveEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_s, CUDA_R_32F, descrA, a_s_val.d_data, CUDA_R_32F, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, f_s.d_data, CUDA_R_32F, x_s.d_data, CUDA_R_32F, CUDA_R_32F);
  cusparseCsrsv_solveEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_d, CUDA_R_64F, descrA, a_d_val.d_data, CUDA_R_64F, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, f_d.d_data, CUDA_R_64F, x_d.d_data, CUDA_R_64F, CUDA_R_64F);
  cusparseCsrsv_solveEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_c, CUDA_C_32F, descrA, a_c_val.d_data, CUDA_C_32F, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, f_c.d_data, CUDA_C_32F, x_c.d_data, CUDA_C_32F, CUDA_C_32F);
  cusparseCsrsv_solveEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_z, CUDA_C_64F, descrA, a_z_val.d_data, CUDA_C_64F, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, f_z.d_data, CUDA_C_64F, x_z.d_data, CUDA_C_64F, CUDA_C_64F);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroySolveAnalysisInfo(info_s);
  cusparseDestroySolveAnalysisInfo(info_d);
  cusparseDestroySolveAnalysisInfo(info_c);
  cusparseDestroySolveAnalysisInfo(info_z);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

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
void test_cusparseTcsr2csc_00() {
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<float2> a_c_val(a_val_vec.data(), 7);
  Data<double2> a_z_val(a_val_vec.data(), 7);
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
  Data<float2> b_c_val(a_val_vec.data(), 7);
  Data<double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

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

  cusparseScsr2csc(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsr2csc(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseCcsr2csc(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseZcsr2csc(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

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

  cudaStreamSynchronize(0);
  cusparseDestroy(handle);

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
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1, 5};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<float2> a_c_val(a_val_vec.data(), 7);
  Data<double2> a_z_val(a_val_vec.data(), 7);
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
  Data<float2> b_c_val(a_val_vec.data(), 7);
  Data<double2> b_z_val(a_val_vec.data(), 7);
  Data<int> b_s_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_d_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_c_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_z_col_ptr(a_row_ptr_vec.data(), 5);
  Data<int> b_s_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_d_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_c_row_ind(a_col_ind_vec.data(), 7);
  Data<int> b_z_row_ind(a_col_ind_vec.data(), 7);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

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

  cusparseScsr2csc(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsr2csc(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseCcsr2csc(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseZcsr2csc(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO);

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

  cudaStreamSynchronize(0);
  cusparseDestroy(handle);

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
  test_cusparseTcsrsv();
  test_cusparseCsrsvEx();
  test_cusparseTcsr2csc_00();
  test_cusparseTcsr2csc_01();

  if (test_passed)
    return 0;
  return -1;
}
