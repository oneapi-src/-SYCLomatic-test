// ===------- cusparse_4.cu -------------------------------- *- CUDA -* ----===//
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

// A * B = C
//
// | 0 1 2 |   | 1 0 0 0 |   | 2 3 10 12 |  
// | 0 0 3 | * | 2 3 0 0 | = | 0 0 15 18 |
// | 4 0 0 |   | 0 0 5 6 |   | 4 0 0  0  |
void test_cusparseSpGEMM() {
  std::vector<float> a_val_vec = {1, 2, 3, 4};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<float2> a_c_val(a_val_vec.data(), 4);
  Data<double2> a_z_val(a_val_vec.data(), 4);
  std::vector<float> a_row_ptr_vec = {0, 2, 3, 4};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {1, 2, 2, 0};
  Data<int> a_col_ind(a_col_ind_vec.data(), 4);

  std::vector<float> b_val_vec = {1, 2, 3, 5, 6};
  Data<float> b_s_val(b_val_vec.data(), 5);
  Data<double> b_d_val(b_val_vec.data(), 5);
  Data<float2> b_c_val(b_val_vec.data(), 5);
  Data<double2> b_z_val(b_val_vec.data(), 5);
  std::vector<float> b_row_ptr_vec = {0, 1, 3, 5};
  Data<int> b_row_ptr(b_row_ptr_vec.data(), 4);
  std::vector<float> b_col_ind_vec = {0, 0, 1, 2, 3};
  Data<int> b_col_ind(b_col_ind_vec.data(), 5);

  float alpha = 1;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

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

  cusparseSpMatDescr_t a_descr_s;
  cusparseSpMatDescr_t a_descr_d;
  cusparseSpMatDescr_t a_descr_c;
  cusparseSpMatDescr_t a_descr_z;
  cusparseCreateCsr(&a_descr_s, 3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&a_descr_d, 3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&a_descr_c, 3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&a_descr_z, 3, 3, 4, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  cusparseSpMatDescr_t b_descr_s;
  cusparseSpMatDescr_t b_descr_d;
  cusparseSpMatDescr_t b_descr_c;
  cusparseSpMatDescr_t b_descr_z;
  cusparseCreateCsr(&b_descr_s, 3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_s_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&b_descr_d, 3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_d_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&b_descr_c, 3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_c_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&b_descr_z, 3, 4, 5, b_row_ptr.d_data, b_col_ind.d_data, b_z_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  Data<int> c_s_row_ptr(4);
  Data<int> c_d_row_ptr(4);
  Data<int> c_c_row_ptr(4);
  Data<int> c_z_row_ptr(4);

  cusparseSpMatDescr_t c_descr_s;
  cusparseSpMatDescr_t c_descr_d;
  cusparseSpMatDescr_t c_descr_c;
  cusparseSpMatDescr_t c_descr_z;
  cusparseCreateCsr(&c_descr_s, 3, 4, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&c_descr_d, 3, 4, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&c_descr_c, 3, 4, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&c_descr_z, 3, 4, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  cusparseSpGEMMDescr_t SpGEMMDescr_s;
  cusparseSpGEMMDescr_t SpGEMMDescr_d;
  cusparseSpGEMMDescr_t SpGEMMDescr_c;
  cusparseSpGEMMDescr_t SpGEMMDescr_z;
  cusparseSpGEMM_createDescr(&SpGEMMDescr_s);
  cusparseSpGEMM_createDescr(&SpGEMMDescr_d);
  cusparseSpGEMM_createDescr(&SpGEMMDescr_c);
  cusparseSpGEMM_createDescr(&SpGEMMDescr_z);

  size_t ws_1_size_s = 0;
  size_t ws_1_size_d = 0;
  size_t ws_1_size_c = 0;
  size_t ws_1_size_z = 0;
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_s, &ws_1_size_s, NULL);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_d, &ws_1_size_d, NULL);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_c, &ws_1_size_c, NULL);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_z, &ws_1_size_z, NULL);

  void *ws_1_s = nullptr;
  void *ws_1_d = nullptr;
  void *ws_1_c = nullptr;
  void *ws_1_z = nullptr;
  cudaMalloc(&ws_1_s, ws_1_size_s);
  cudaMalloc(&ws_1_d, ws_1_size_d);
  cudaMalloc(&ws_1_c, ws_1_size_c);
  cudaMalloc(&ws_1_z, ws_1_size_z);

  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_s, &ws_1_size_s, ws_1_s);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_d, &ws_1_size_d, ws_1_d);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_c, &ws_1_size_c, ws_1_c);
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_z, &ws_1_size_z, ws_1_z);

  size_t ws_2_size_s = 0;
  size_t ws_2_size_d = 0;
  size_t ws_2_size_c = 0;
  size_t ws_2_size_z = 0;
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_s, &ws_2_size_s, NULL);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_d, &ws_2_size_d, NULL);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_c, &ws_2_size_c, NULL);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_z, &ws_2_size_z, NULL);

  void *ws_2_s = nullptr;
  void *ws_2_d = nullptr;
  void *ws_2_c = nullptr;
  void *ws_2_z = nullptr;
  cudaMalloc(&ws_2_s, ws_2_size_s);
  cudaMalloc(&ws_2_d, ws_2_size_d);
  cudaMalloc(&ws_2_c, ws_2_size_c);
  cudaMalloc(&ws_2_z, ws_2_size_z);

  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_s, &ws_2_size_s, ws_2_s);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_d, &ws_2_size_d, ws_2_d);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_c, &ws_2_size_c, ws_2_c);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_z, &ws_2_size_z, ws_2_z);

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
  cusparseSpMatGetSize(c_descr_s, &c_row_s, &c_col_s, &c_nnz_s);
  cusparseSpMatGetSize(c_descr_d, &c_row_d, &c_col_d, &c_nnz_d);
  cusparseSpMatGetSize(c_descr_c, &c_row_c, &c_col_c, &c_nnz_c);
  cusparseSpMatGetSize(c_descr_z, &c_row_z, &c_col_z, &c_nnz_z);

  Data<float> c_s_val(c_nnz_s);
  Data<double> c_d_val(c_nnz_d);
  Data<float2> c_c_val(c_nnz_c);
  Data<double2> c_z_val(c_nnz_z);
  Data<int> c_s_col_ind(c_nnz_s);
  Data<int> c_d_col_ind(c_nnz_d);
  Data<int> c_c_col_ind(c_nnz_c);
  Data<int> c_z_col_ind(c_nnz_z);

  cusparseCsrSetPointers(c_descr_s, c_s_row_ptr.d_data, c_s_col_ind.d_data, c_s_val.d_data);
  cusparseCsrSetPointers(c_descr_d, c_d_row_ptr.d_data, c_d_col_ind.d_data, c_d_val.d_data);
  cusparseCsrSetPointers(c_descr_c, c_c_row_ptr.d_data, c_c_col_ind.d_data, c_c_val.d_data);
  cusparseCsrSetPointers(c_descr_z, c_z_row_ptr.d_data, c_z_col_ind.d_data, c_z_val.d_data);

  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_s);
  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_d);
  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_c);
  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescr_z);

  cudaStreamSynchronize(0);

  cudaFree(ws_1_s);
  cudaFree(ws_1_d);
  cudaFree(ws_1_c);
  cudaFree(ws_1_z);
  cudaFree(ws_2_s);
  cudaFree(ws_2_d);
  cudaFree(ws_2_c);
  cudaFree(ws_2_z);
  cusparseDestroySpMat(a_descr_s);
  cusparseDestroySpMat(a_descr_d);
  cusparseDestroySpMat(a_descr_c);
  cusparseDestroySpMat(a_descr_z);
  cusparseDestroySpMat(b_descr_s);
  cusparseDestroySpMat(b_descr_d);
  cusparseDestroySpMat(b_descr_c);
  cusparseDestroySpMat(b_descr_z);
  cusparseDestroySpMat(c_descr_s);
  cusparseDestroySpMat(c_descr_d);
  cusparseDestroySpMat(c_descr_c);
  cusparseDestroySpMat(c_descr_z);
  cusparseSpGEMM_destroyDescr(SpGEMMDescr_s);
  cusparseSpGEMM_destroyDescr(SpGEMMDescr_d);
  cusparseSpGEMM_destroyDescr(SpGEMMDescr_c);
  cusparseSpGEMM_destroyDescr(SpGEMMDescr_z);
  cusparseDestroy(handle);

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

int main() {
  test_cusparseSpGEMM();

  if (test_passed)
    return 0;
  return -1;
}
