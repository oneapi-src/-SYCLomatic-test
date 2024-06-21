// ===------- cusparse_3.cu -------------------------------- *- CUDA -* ----===//
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

void test_cusparseSpMV() {
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<float2> b_c(b_vec.data(), 5);
  Data<double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
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

  cusparseSpMatDescr_t a_descr_s;
  cusparseSpMatDescr_t a_descr_d;
  cusparseSpMatDescr_t a_descr_c;
  cusparseSpMatDescr_t a_descr_z;
  cusparseCreateCsr(&a_descr_s, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&a_descr_d, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&a_descr_c, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&a_descr_z, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  cusparseDnVecDescr_t b_descr_s;
  cusparseDnVecDescr_t b_descr_d;
  cusparseDnVecDescr_t b_descr_c;
  cusparseDnVecDescr_t b_descr_z;
  cusparseCreateDnVec(&b_descr_s, 5, b_s.d_data, CUDA_R_32F);
  cusparseCreateDnVec(&b_descr_d, 5, b_d.d_data, CUDA_R_64F);
  cusparseCreateDnVec(&b_descr_c, 5, b_c.d_data, CUDA_C_32F);
  cusparseCreateDnVec(&b_descr_z, 5, b_z.d_data, CUDA_C_64F);

  cusparseDnVecDescr_t c_descr_s;
  cusparseDnVecDescr_t c_descr_d;
  cusparseDnVecDescr_t c_descr_c;
  cusparseDnVecDescr_t c_descr_z;
  cusparseCreateDnVec(&c_descr_s, 4, c_s.d_data, CUDA_R_32F);
  cusparseCreateDnVec(&c_descr_d, 4, c_d.d_data, CUDA_R_64F);
  cusparseCreateDnVec(&c_descr_c, 4, c_c.d_data, CUDA_C_32F);
  cusparseCreateDnVec(&c_descr_z, 4, c_z.d_data, CUDA_C_64F);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size_s);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size_d);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size_c);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size_z);

  void *ws_s = nullptr;
  void *ws_d = nullptr;
  void *ws_c = nullptr;
  void *ws_z = nullptr;
  cudaMalloc(&ws_s, ws_size_s);
  cudaMalloc(&ws_d, ws_size_d);
  cudaMalloc(&ws_c, ws_size_c);
  cudaMalloc(&ws_z, ws_size_z);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, ws_s);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ws_d);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, ws_c);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, ws_z);

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);

  cudaFree(ws_s);
  cudaFree(ws_d);
  cudaFree(ws_c);
  cudaFree(ws_z);
  cusparseDestroySpMat(a_descr_s);
  cusparseDestroySpMat(a_descr_d);
  cusparseDestroySpMat(a_descr_c);
  cusparseDestroySpMat(a_descr_z);
  cusparseDestroyDnVec(b_descr_s);
  cusparseDestroyDnVec(b_descr_d);
  cusparseDestroyDnVec(b_descr_c);
  cusparseDestroyDnVec(b_descr_z);
  cusparseDestroyDnVec(c_descr_s);
  cusparseDestroyDnVec(c_descr_d);
  cusparseDestroyDnVec(c_descr_c);
  cusparseDestroyDnVec(c_descr_z);
  cusparseDestroy(handle);

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
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Data<float> b_s(b_vec.data(), 10);
  Data<double> b_d(b_vec.data(), 10);
  Data<float2> b_c(b_vec.data(), 10);
  Data<double2> b_z(b_vec.data(), 10);

  Data<float> c_s(8);
  Data<double> c_d(8);
  Data<float2> c_c(8);
  Data<double2> c_z(8);

  float alpha = 10;
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

  cusparseSpMatDescr_t a_descr_s;
  cusparseSpMatDescr_t a_descr_d;
  cusparseSpMatDescr_t a_descr_c;
  cusparseSpMatDescr_t a_descr_z;
  cusparseCreateCsr(&a_descr_s, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&a_descr_d, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&a_descr_c, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&a_descr_z, 4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  cusparseDnMatDescr_t b_descr_s;
  cusparseDnMatDescr_t b_descr_d;
  cusparseDnMatDescr_t b_descr_c;
  cusparseDnMatDescr_t b_descr_z;
  cusparseCreateDnMat(&b_descr_s, 5, 2, 5, b_s.d_data, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&b_descr_d, 5, 2, 5, b_d.d_data, CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&b_descr_c, 5, 2, 5, b_c.d_data, CUDA_C_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&b_descr_z, 5, 2, 5, b_z.d_data, CUDA_C_64F, CUSPARSE_ORDER_COL);

  cusparseDnMatDescr_t c_descr_s;
  cusparseDnMatDescr_t c_descr_d;
  cusparseDnMatDescr_t c_descr_c;
  cusparseDnMatDescr_t c_descr_z;
  cusparseCreateDnMat(&c_descr_s, 4, 2, 4, c_s.d_data, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&c_descr_d, 4, 2, 4, c_d.d_data, CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&c_descr_c, 4, 2, 4, c_c.d_data, CUDA_C_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&c_descr_z, 4, 2, 4, c_z.d_data, CUDA_C_64F, CUSPARSE_ORDER_COL);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size_s);
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size_d);
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size_c);
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size_z);

  void *ws_s = nullptr;
  void *ws_d = nullptr;
  void *ws_c = nullptr;
  void *ws_z = nullptr;
  cudaMalloc(&ws_s, ws_size_s);
  cudaMalloc(&ws_d, ws_size_d);
  cudaMalloc(&ws_c, ws_size_c);
  cudaMalloc(&ws_z, ws_size_z);

  cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws_s);
  cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws_d);
  cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws_c);
  cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws_z);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_s.d_data, a_descr_s, b_descr_s, beta_s.d_data, c_descr_s, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws_s);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d.d_data, a_descr_d, b_descr_d, beta_d.d_data, c_descr_d, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws_d);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data, c_descr_c, CUDA_C_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws_c);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data, c_descr_z, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws_z);

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);

  cudaFree(ws_s);
  cudaFree(ws_d);
  cudaFree(ws_c);
  cudaFree(ws_z);
  cusparseDestroySpMat(a_descr_s);
  cusparseDestroySpMat(a_descr_d);
  cusparseDestroySpMat(a_descr_c);
  cusparseDestroySpMat(a_descr_z);
  cusparseDestroyDnMat(b_descr_s);
  cusparseDestroyDnMat(b_descr_d);
  cusparseDestroyDnMat(b_descr_c);
  cusparseDestroyDnMat(b_descr_z);
  cusparseDestroyDnMat(c_descr_s);
  cusparseDestroyDnMat(c_descr_d);
  cusparseDestroyDnMat(c_descr_c);
  cusparseDestroyDnMat(c_descr_z);
  cusparseDestroy(handle);

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

int main() {
  test_cusparseSpMV();
  test_cusparseSpMM();

  if (test_passed)
    return 0;
  return -1;
}
