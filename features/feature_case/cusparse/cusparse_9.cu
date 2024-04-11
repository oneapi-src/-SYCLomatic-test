// ===------- cusparse_9.cu -------------------------------- *- CUDA -* ----===//
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

// op(A) * C = alpha * op(B)
//
// | 1 1 2 |   | 1 4 |   | 9  21 |  
// | 0 1 3 | * | 2 5 | = | 11 23 |
// | 0 0 1 |   | 3 6 |   | 3  6  |
void test_cusparseSpSM() {
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

  std::vector<float> b_vec = {9, 11, 3, 21, 23, 6};
  Data<float> b_s(b_vec.data(), 6);
  Data<double> b_d(b_vec.data(), 6);
  Data<float2> b_c(b_vec.data(), 6);
  Data<double2> b_z(b_vec.data(), 6);

  Data<float> c_s(6);
  Data<double> c_d(6);
  Data<float2> c_c(6);
  Data<double2> c_z(6);

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
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  cusparseHandle_t handle;
  cusparseSpMatDescr_t matA_s, matA_d, matA_c, matA_z;
  cusparseDnMatDescr_t matB_s, matB_d, matB_c, matB_z;
  cusparseDnMatDescr_t matC_s, matC_d, matC_c, matC_z;
  cusparseCreate(&handle);
  cusparseCreateCsr(&matA_s, 3, 3, 6, a_row_ptr_s.d_data, a_col_ind_s.d_data, a_s_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matA_d, 3, 3, 6, a_row_ptr_d.d_data, a_col_ind_d.d_data, a_d_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateCsr(&matA_c, 3, 3, 6, a_row_ptr_c.d_data, a_col_ind_c.d_data, a_c_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateCsr(&matA_z, 3, 3, 6, a_row_ptr_z.d_data, a_col_ind_z.d_data, a_z_val.d_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

  cusparseCreateDnMat(&matB_s, 3, 2, 3, b_s.d_data, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matB_d, 3, 2, 3, b_d.d_data, CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matB_c, 3, 2, 3, b_c.d_data, CUDA_C_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matB_z, 3, 2, 3, b_z.d_data, CUDA_C_64F, CUSPARSE_ORDER_COL);

  cusparseCreateDnMat(&matC_s, 3, 2, 3, c_s.d_data, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matC_d, 3, 2, 3, c_d.d_data, CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matC_c, 3, 2, 3, c_c.d_data, CUDA_C_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&matC_z, 3, 2, 3, c_z.d_data, CUDA_C_64F, CUSPARSE_ORDER_COL);

  cusparseSpSMDescr_t spsmDescr_s;
  cusparseSpSMDescr_t spsmDescr_d;
  cusparseSpSMDescr_t spsmDescr_c;
  cusparseSpSMDescr_t spsmDescr_z;
  cusparseSpSM_createDescr(&spsmDescr_s);
  cusparseSpSM_createDescr(&spsmDescr_d);
  cusparseSpSM_createDescr(&spsmDescr_c);
  cusparseSpSM_createDescr(&spsmDescr_z);

  cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
  cusparseSpMatSetAttribute(matA_s, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
  cusparseSpMatSetAttribute(matA_d, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
  cusparseSpMatSetAttribute(matA_c, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
  cusparseSpMatSetAttribute(matA_z, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseSpMatSetAttribute(matA_s, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
  cusparseSpMatSetAttribute(matA_d, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
  cusparseSpMatSetAttribute(matA_c, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
  cusparseSpMatSetAttribute(matA_z, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));

  float alpha_s = 1.0f;
  double alpha_d = 1.0f;
  float2 alpha_c = {1.0f, 0.0f};
  double2 alpha_z = {1.0f, 0.0f};
  size_t bufferSize_s = 0;
  size_t bufferSize_d = 0;
  size_t bufferSize_c = 0;
  size_t bufferSize_z = 0;
  cusparseSpSM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_s, matA_s, matB_s, matC_s, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_s, &bufferSize_s);
  cusparseSpSM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_d, matA_d, matB_d, matC_d, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_d, &bufferSize_d);
  cusparseSpSM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_c, matA_c, matB_c, matC_c, CUDA_C_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_c, &bufferSize_c);
  cusparseSpSM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_z, matA_z, matB_z, matC_z, CUDA_C_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_z, &bufferSize_z);

  void* dBuffer_s = NULL;
  void* dBuffer_d = NULL;
  void* dBuffer_c = NULL;
  void* dBuffer_z = NULL;
  cudaMalloc(&dBuffer_s, bufferSize_s);
  cudaMalloc(&dBuffer_d, bufferSize_d);
  cudaMalloc(&dBuffer_c, bufferSize_c);
  cudaMalloc(&dBuffer_z, bufferSize_z);

  cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_s, matA_s, matB_s, matC_s, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_s, &bufferSize_s);
  cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_d, matA_d, matB_d, matC_d, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_d, &bufferSize_d);
  cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_c, matA_c, matB_c, matC_c, CUDA_C_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_c, &bufferSize_c);
  cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_z, matA_z, matB_z, matC_z, CUDA_C_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_z, &bufferSize_z);

  cusparseSpSM_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_s, matA_s, matB_s, matC_s, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_s);
  cusparseSpSM_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_d, matA_d, matB_d, matC_d, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_d);
  cusparseSpSM_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_c, matA_c, matB_c, matC_c, CUDA_C_32F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_c);
  cusparseSpSM_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_z, matA_z, matB_z, matC_z, CUDA_C_64F, CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr_z);

  cusparseDestroySpMat(matA_s);
  cusparseDestroySpMat(matA_d);
  cusparseDestroySpMat(matA_c);
  cusparseDestroySpMat(matA_z);
  cusparseDestroyDnMat(matB_s);
  cusparseDestroyDnMat(matB_d);
  cusparseDestroyDnMat(matB_c);
  cusparseDestroyDnMat(matB_z);
  cusparseDestroyDnMat(matC_s);
  cusparseDestroyDnMat(matC_d);
  cusparseDestroyDnMat(matC_c);
  cusparseDestroyDnMat(matC_z);
  cusparseSpSM_destroyDescr(spsmDescr_s);
  cusparseSpSM_destroyDescr(spsmDescr_d);
  cusparseSpSM_destroyDescr(spsmDescr_c);
  cusparseSpSM_destroyDescr(spsmDescr_z);
  cusparseDestroy(handle);

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaFree(dBuffer_s);
  cudaFree(dBuffer_d);
  cudaFree(dBuffer_c);
  cudaFree(dBuffer_z);

  float expect_c[6] = {1, 2, 3, 4, 5, 6};
  if (compare_result(expect_c, c_s.h_data, 6) &&
      compare_result(expect_c, c_d.h_data, 6) &&
      compare_result(expect_c, c_c.h_data, 6) &&
      compare_result(expect_c, c_z.h_data, 6))
    printf("SpSM pass\n");
  else {
    printf("SpSM fail\n");
    test_passed = false;
    printf("%f,%f,%f,%f,%f,%f\n", c_s.h_data[0], c_s.h_data[1], c_s.h_data[2],
           c_s.h_data[3], c_s.h_data[4], c_s.h_data[5]);
    printf("%f,%f,%f,%f,%f,%f\n", c_d.h_data[0], c_d.h_data[1], c_d.h_data[2],
           c_d.h_data[3], c_d.h_data[4], c_d.h_data[5]);
    printf("%f,%f,%f,%f,%f,%f\n", c_c.h_data[0], c_c.h_data[1], c_c.h_data[2],
           c_c.h_data[3], c_c.h_data[4], c_c.h_data[5]);
    printf("%f,%f,%f,%f,%f,%f\n", c_z.h_data[0], c_z.h_data[1], c_z.h_data[2],
           c_z.h_data[3], c_z.h_data[4], c_z.h_data[5]);
  }
}

int main() {
  test_cusparseSpSM();

  if (test_passed)
    return 0;
  return -1;
}
