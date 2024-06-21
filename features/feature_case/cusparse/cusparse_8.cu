// ===------- cusparse_8.cu -------------------------------- *- CUDA -* ----===//
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

// | 1 1 2 0 |
// | 0 1 3 0 |
// | 0 0 1 5 |
void test_cusparseCsr2csc_00() {
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


  size_t ws_size_s = 0;
  size_t ws_size_d = 0;
  size_t ws_size_c = 0;
  size_t ws_size_z = 0;
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_s);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_d);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUDA_C_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_c);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUDA_C_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_z);

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  cudaMalloc(&ws_s, ws_size_s);
  cudaMalloc(&ws_d, ws_size_d);
  cudaMalloc(&ws_c, ws_size_c);
  cudaMalloc(&ws_z, ws_size_z);

  cusparseCsr2cscEx2(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_s);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_d);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUDA_C_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_c);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUDA_C_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_z);

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

  cudaFree(ws_s);
  cudaFree(ws_d);
  cudaFree(ws_c);
  cudaFree(ws_z);
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


  size_t ws_size_s = 0;
  size_t ws_size_d = 0;
  size_t ws_size_c = 0;
  size_t ws_size_z = 0;
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_s);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUDA_R_64F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_d);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUDA_C_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_c);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUDA_C_64F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size_z);

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  cudaMalloc(&ws_s, ws_size_s);
  cudaMalloc(&ws_d, ws_size_d);
  cudaMalloc(&ws_c, ws_size_c);
  cudaMalloc(&ws_z, ws_size_z);

  cusparseCsr2cscEx2(handle, 3, 4, 7, a_s_val.d_data, a_s_row_ptr.d_data, a_s_col_ind.d_data, b_s_val.d_data, b_s_col_ptr.d_data, b_s_row_ind.d_data, CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_s);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_d_val.d_data, a_d_row_ptr.d_data, a_d_col_ind.d_data, b_d_val.d_data, b_d_col_ptr.d_data, b_d_row_ind.d_data, CUDA_R_64F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_d);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_c_val.d_data, a_c_row_ptr.d_data, a_c_col_ind.d_data, b_c_val.d_data, b_c_col_ptr.d_data, b_c_row_ind.d_data, CUDA_C_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_c);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_z_val.d_data, a_z_row_ptr.d_data, a_z_col_ind.d_data, b_z_val.d_data, b_z_col_ptr.d_data, b_z_row_ind.d_data, CUDA_C_64F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws_z);

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

  cudaFree(ws_s);
  cudaFree(ws_d);
  cudaFree(ws_c);
  cudaFree(ws_z);
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
    printf("Csr2csc 01 pass\n");
  else {
    printf("Csr2csc 01 fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseCsr2csc_00();
  test_cusparseCsr2csc_01();

  if (test_passed)
    return 0;
  return -1;
}
