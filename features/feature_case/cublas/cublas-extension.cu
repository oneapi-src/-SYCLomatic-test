// ====------ cublas-extension.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>
#include <cstdint>

template<class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float* input_data, int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num, cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num, cudaMemcpyDeviceToHost);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }
private:
  inline void from_float_convert(float* in, d_data_t* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t* in, float* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<float2>::from_float_convert(float* in, float2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float* in, double2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<std::complex<int8_t>>::from_float_convert(float* in, std::complex<int8_t>* out) {
  for (int i = 0; i < element_num; i++)
    reinterpret_cast<int8_t(&)[2]>(out[i])[0] = int8_t(in[i]);
}

template <>
inline void Data<float2>::to_float_convert(float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<half>::to_float_convert(half* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = __half2float(in[i]);
}
template <>
inline void Data<__nv_bfloat16>::to_float_convert(__nv_bfloat16* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i];
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.1) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cublasNrm2Ex() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<float2> x_f2(v.data(), 4);
  Data<double2> x_d2(v.data(), 4);
  Data<half> x_h(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<float> res_f2(1);
  Data<double> res_d2(1);
  Data<half> res_h(1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();
  x_h.H2D();

  cublasNrm2Ex(handle, 4, x_f.d_data, CUDA_R_32F, 1, res_f.d_data, CUDA_R_32F, CUDA_R_32F);
  cublasNrm2Ex(handle, 4, x_d.d_data, CUDA_R_64F, 1, res_d.d_data, CUDA_R_64F, CUDA_R_64F);
  cublasNrm2Ex(handle, 4, x_f2.d_data, CUDA_C_32F, 1, res_f2.d_data, CUDA_R_32F, CUDA_R_32F);
  cublasNrm2Ex(handle, 4, x_d2.d_data, CUDA_C_64F, 1, res_d2.d_data, CUDA_R_64F, CUDA_R_64F);
  cublasNrm2Ex(handle, 4, x_h.d_data, CUDA_R_16F, 1, res_h.d_data, CUDA_R_16F, CUDA_R_32F);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();
  res_h.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect = 9.32738;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1)
      && compare_result(&expect, res_h.h_data, 1))
    printf("Nrm2Ex pass\n");
  else {
    printf("Nrm2Ex fail\n");
    test_passed = false;
  }
}

void test_cublasDotEx() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<float2> x_f2(v.data(), 4);
  Data<double2> x_d2(v.data(), 4);
  Data<half> x_h(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<float2> y_f2(v.data(), 4);
  Data<double2> y_d2(v.data(), 4);
  Data<half> y_h(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<float2> res_f2(1);
  Data<double2> res_d2(1);
  Data<half> res_h(1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();
  x_h.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();
  y_h.H2D();

  cublasDotEx(handle, 4, x_f.d_data, CUDA_R_32F, 1, y_f.d_data, CUDA_R_32F, 1, res_f.d_data, CUDA_R_32F, CUDA_R_32F);
  cublasDotEx(handle, 4, x_d.d_data, CUDA_R_64F, 1, y_d.d_data, CUDA_R_64F, 1, res_d.d_data, CUDA_R_64F, CUDA_R_64F);
  cublasDotEx(handle, 4, x_f2.d_data, CUDA_C_32F, 1, y_f2.d_data, CUDA_C_32F, 1, res_f2.d_data, CUDA_C_32F, CUDA_C_32F);
  cublasDotEx(handle, 4, x_d2.d_data, CUDA_C_64F, 1, y_d2.d_data, CUDA_C_64F, 1, res_d2.d_data, CUDA_C_64F, CUDA_C_64F);
  cublasDotEx(handle, 4, x_h.d_data, CUDA_R_16F, 1, y_h.d_data, CUDA_R_16F, 1, res_h.d_data, CUDA_R_16F, CUDA_R_32F);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();
  res_h.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect = 87;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1)
      && compare_result(&expect, res_h.h_data, 1))
    printf("DotEx pass\n");
  else {
    printf("DotEx fail\n");
    test_passed = false;
  }
}

void test_cublasDotcEx() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<float2> x_f2(v.data(), 4);
  Data<double2> x_d2(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<float2> y_f2(v.data(), 4);
  Data<double2> y_d2(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<float2> res_f2(1);
  Data<double2> res_d2(1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();

  cublasDotcEx(handle, 4, x_f.d_data, CUDA_R_32F, 1, y_f.d_data, CUDA_R_32F, 1, res_f.d_data, CUDA_R_32F, CUDA_R_32F);
  cublasDotcEx(handle, 4, x_d.d_data, CUDA_R_64F, 1, y_d.d_data, CUDA_R_64F, 1, res_d.d_data, CUDA_R_64F, CUDA_R_64F);
  cublasDotcEx(handle, 4, x_f2.d_data, CUDA_C_32F, 1, y_f2.d_data, CUDA_C_32F, 1, res_f2.d_data, CUDA_C_32F, CUDA_C_32F);
  cublasDotcEx(handle, 4, x_d2.d_data, CUDA_C_64F, 1, y_d2.d_data, CUDA_C_64F, 1, res_d2.d_data, CUDA_C_64F, CUDA_C_64F);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect = 87;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1))
    printf("DotcEx pass\n");
  else {
    printf("DotcEx fail\n");
    test_passed = false;
  }
}

void test_cublasScalEx() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<float2> x_f2(v.data(), 4);
  Data<double2> x_d2(v.data(), 4);
  Data<half> x_h(v.data(), 4);

  float alpha = 10;
  Data<float> alpha_f(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_f2(&alpha, 1);
  Data<double2> alpha_d2(&alpha, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();
  x_h.H2D();

  alpha_f.H2D();
  alpha_d.H2D();
  alpha_f2.H2D();
  alpha_d2.H2D();

  cublasScalEx(handle, 4, alpha_f.d_data, CUDA_R_32F, x_f.d_data, CUDA_R_32F, 1, CUDA_R_32F);
  cublasScalEx(handle, 4, alpha_d.d_data, CUDA_R_64F, x_d.d_data, CUDA_R_64F, 1, CUDA_R_64F);
  cublasScalEx(handle, 4, alpha_f2.d_data, CUDA_C_32F, x_f2.d_data, CUDA_C_32F, 1, CUDA_C_32F);
  cublasScalEx(handle, 4, alpha_d2.d_data, CUDA_C_64F, x_d2.d_data, CUDA_C_64F, 1, CUDA_C_64F);
  cublasScalEx(handle, 4, alpha_f.d_data, CUDA_R_32F, x_h.d_data, CUDA_R_16F, 1, CUDA_R_32F);

  x_f.D2H();
  x_d.D2H();
  x_f2.D2H();
  x_d2.D2H();
  x_h.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = {20, 30, 50, 70};
  if (compare_result(expect, x_f.h_data, 4)
      && compare_result(expect, x_d.h_data, 4)
      && compare_result(expect, x_f2.h_data, 4)
      && compare_result(expect, x_d2.h_data, 4)
      && compare_result(expect, x_h.h_data, 4))
    printf("ScalEx pass\n");
  else {
    printf("ScalEx fail\n");
    test_passed = false;
  }
}

void test_cublasAxpyEx() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<float2> x_f2(v.data(), 4);
  Data<double2> x_d2(v.data(), 4);
  Data<half> x_h(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<float2> y_f2(v.data(), 4);
  Data<double2> y_d2(v.data(), 4);
  Data<half> y_h(v.data(), 4);

  float alpha = 10;
  Data<float> alpha_f(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_f2(&alpha, 1);
  Data<double2> alpha_d2(&alpha, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();
  x_h.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();
  y_h.H2D();

  alpha_f.H2D();
  alpha_d.H2D();
  alpha_f2.H2D();
  alpha_d2.H2D();

  cublasAxpyEx(handle, 4, alpha_f.d_data, CUDA_R_32F, x_f.d_data, CUDA_R_32F, 1, y_f.d_data, CUDA_R_32F, 1, CUDA_R_32F);
  cublasAxpyEx(handle, 4, alpha_d.d_data, CUDA_R_64F, x_d.d_data, CUDA_R_64F, 1, y_d.d_data, CUDA_R_64F, 1, CUDA_R_64F);
  cublasAxpyEx(handle, 4, alpha_f2.d_data, CUDA_C_32F, x_f2.d_data, CUDA_C_32F, 1, y_f2.d_data, CUDA_C_32F, 1, CUDA_C_32F);
  cublasAxpyEx(handle, 4, alpha_d2.d_data, CUDA_C_64F, x_d2.d_data, CUDA_C_64F, 1, y_d2.d_data, CUDA_C_64F, 1, CUDA_C_64F);
  cublasAxpyEx(handle, 4, alpha_f.d_data, CUDA_R_32F, x_h.d_data, CUDA_R_16F, 1, y_h.d_data, CUDA_R_16F, 1, CUDA_R_32F);

  y_f.D2H();
  y_d.D2H();
  y_f2.D2H();
  y_d2.D2H();
  y_h.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = {22, 33, 55, 77};
  if (compare_result(expect, y_f.h_data, 4)
      && compare_result(expect, y_d.h_data, 4)
      && compare_result(expect, y_f2.h_data, 4)
      && compare_result(expect, y_d2.h_data, 4)
      && compare_result(expect, y_h.h_data, 4))
    printf("AxpyEx pass\n");
  else {
    printf("AxpyEx fail\n");
    test_passed = false;
  }
}

void test_cublasRotEx() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<__nv_bfloat16> x0(v.data(), 4);
  Data<half>          x1(v.data(), 4);
  Data<float>         x2(v.data(), 4);
  Data<double>        x3(v.data(), 4);
  Data<float2>        x4(v.data(), 4);
  Data<float2>        x5(v.data(), 4);
  Data<double2>       x6(v.data(), 4);
  Data<double2>       x7(v.data(), 4);

  Data<__nv_bfloat16> y0(v.data(), 4);
  Data<half>          y1(v.data(), 4);
  Data<float>         y2(v.data(), 4);
  Data<double>        y3(v.data(), 4);
  Data<float2>        y4(v.data(), 4);
  Data<float2>        y5(v.data(), 4);
  Data<double2>       y6(v.data(), 4);
  Data<double2>       y7(v.data(), 4);

  float cos = 0.866;
  float sin = 0.5;
  Data<__nv_bfloat16> cos0(&cos, 1);
  Data<half>          cos1(&cos, 1);
  Data<float>         cos2(&cos, 1);
  Data<double>        cos3(&cos, 1);
  Data<float>         cos4(&cos, 1);
  Data<float2>        cos5(&cos, 1);
  Data<double>        cos6(&cos, 1);
  Data<double2>       cos7(&cos, 1);

  Data<__nv_bfloat16> sin0(&sin, 1);
  Data<half>          sin1(&sin, 1);
  Data<float>         sin2(&sin, 1);
  Data<double>        sin3(&sin, 1);
  Data<float>         sin4(&sin, 1);
  Data<float2>        sin5(&sin, 1);
  Data<double>        sin6(&sin, 1);
  Data<double2>       sin7(&sin, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x0.H2D();
  x1.H2D();
  x2.H2D();
  x3.H2D();
  x4.H2D();
  x5.H2D();
  x6.H2D();
  x7.H2D();

  y0.H2D();
  y1.H2D();
  y2.H2D();
  y3.H2D();
  y4.H2D();
  y5.H2D();
  y6.H2D();
  y7.H2D();

  sin0.H2D();
  sin1.H2D();
  sin2.H2D();
  sin3.H2D();
  sin4.H2D();
  sin5.H2D();
  sin6.H2D();
  sin7.H2D();

  cos0.H2D();
  cos1.H2D();
  cos2.H2D();
  cos3.H2D();
  cos4.H2D();
  cos5.H2D();
  cos6.H2D();
  cos7.H2D();

  cublasRotEx(handle, 4, x0.d_data, CUDA_R_16BF, 1,  y0.d_data, CUDA_R_16BF, 1,  cos0.d_data, sin0.d_data, CUDA_R_16BF, CUDA_R_32F);
  cublasRotEx(handle, 4, x1.d_data, CUDA_R_16F, 1,  y1.d_data, CUDA_R_16F, 1,  cos1.d_data, sin1.d_data, CUDA_R_16F, CUDA_R_32F);
  cublasRotEx(handle, 4, x2.d_data, CUDA_R_32F, 1,  y2.d_data, CUDA_R_32F, 1,  cos2.d_data, sin2.d_data, CUDA_R_32F, CUDA_R_32F);
  cublasRotEx(handle, 4, x3.d_data, CUDA_R_64F, 1,  y3.d_data, CUDA_R_64F, 1,  cos3.d_data, sin3.d_data, CUDA_R_64F, CUDA_R_64F);
  cublasRotEx(handle, 4, x4.d_data, CUDA_C_32F, 1,  y4.d_data, CUDA_C_32F, 1,  cos4.d_data, sin4.d_data, CUDA_R_32F, CUDA_C_32F);
  cublasRotEx(handle, 4, x5.d_data, CUDA_C_32F, 1,  y5.d_data, CUDA_C_32F, 1,  cos5.d_data, sin5.d_data, CUDA_C_32F, CUDA_C_32F);
  cublasRotEx(handle, 4, x6.d_data, CUDA_C_64F, 1,  y6.d_data, CUDA_C_64F, 1,  cos6.d_data, sin6.d_data, CUDA_R_64F, CUDA_C_64F);
  cublasRotEx(handle, 4, x7.d_data, CUDA_C_64F, 1,  y7.d_data, CUDA_C_64F, 1,  cos7.d_data, sin7.d_data, CUDA_C_64F, CUDA_C_64F);

  x0.D2H();
  x1.D2H();
  x2.D2H();
  x3.D2H();
  x4.D2H();
  x5.D2H();
  x6.D2H();
  x7.D2H();

  y0.D2H();
  y1.D2H();
  y2.D2H();
  y3.D2H();
  y4.D2H();
  y5.D2H();
  y6.D2H();
  y7.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect_x[4] = {2.732000,4.098000,6.830000,9.562000};
  float expect_y[4] = {0.732000,1.098000,1.830000,2.562000};
  if (compare_result(expect_x, x0.h_data, 4) &&
      compare_result(expect_x, x1.h_data, 4) &&
      compare_result(expect_x, x2.h_data, 4) &&
      compare_result(expect_x, x3.h_data, 4) &&
      compare_result(expect_x, x4.h_data, 4) &&
      compare_result(expect_x, x5.h_data, 4) &&
      compare_result(expect_x, x6.h_data, 4) &&
      compare_result(expect_x, x7.h_data, 4) &&
      compare_result(expect_y, y0.h_data, 4) &&
      compare_result(expect_y, y1.h_data, 4) &&
      compare_result(expect_y, y2.h_data, 4) &&
      compare_result(expect_y, y3.h_data, 4) &&
      compare_result(expect_y, y4.h_data, 4) &&
      compare_result(expect_y, y5.h_data, 4) &&
      compare_result(expect_y, y6.h_data, 4) &&
      compare_result(expect_y, y7.h_data, 4))
    printf("RotEx pass\n");
  else {
    printf("RotEx fail\n");
    test_passed = false;
  }
}

void test_cublasGemmEx() {
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<half> a0(v.data(), 16);
  Data<std::int8_t> a1(v.data(), 16);
  Data<__nv_bfloat16> a2(v.data(), 16);
  Data<half> a3(v.data(), 16);
  Data<std::int8_t> a4(v.data(), 16);
  Data<__nv_bfloat16> a5(v.data(), 16);
  Data<half> a6(v.data(), 16);
  Data<float> a7(v.data(), 16);
  Data<float2> a9(v.data(), 16);
  Data<double> a10(v.data(), 16);
  Data<double2> a11(v.data(), 16);

  Data<half> b0(v.data(), 16);
  Data<std::int8_t> b1(v.data(), 16);
  Data<__nv_bfloat16> b2(v.data(), 16);
  Data<half> b3(v.data(), 16);
  Data<std::int8_t> b4(v.data(), 16);
  Data<__nv_bfloat16> b5(v.data(), 16);
  Data<half> b6(v.data(), 16);
  Data<float> b7(v.data(), 16);
  Data<float2> b9(v.data(), 16);
  Data<double> b10(v.data(), 16);
  Data<double2> b11(v.data(), 16);

  Data<half> c0(16);
  Data<std::int32_t> c1(16);
  Data<__nv_bfloat16> c2(16);
  Data<half> c3(16);
  Data<float> c4(16);
  Data<float> c5(16);
  Data<float> c6(16);
  Data<float> c7(16);
  Data<float2> c9(16);
  Data<double> c10(16);
  Data<double2> c11(16);

  float alpha = 2;
  float beta = 0;
  Data<half> alpha0(&alpha, 1);
  Data<std::int32_t> alpha1(&alpha, 1);
  Data<float> alpha2(&alpha, 1);
  Data<float> alpha3(&alpha, 1);
  Data<float> alpha4(&alpha, 1);
  Data<float> alpha5(&alpha, 1);
  Data<float> alpha6(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<double2> alpha11(&alpha, 1);

  Data<half> beta0(&beta, 1);
  Data<std::int32_t> beta1(&beta, 1);
  Data<float> beta2(&beta, 1);
  Data<float> beta3(&beta, 1);
  Data<float> beta4(&beta, 1);
  Data<float> beta5(&beta, 1);
  Data<float> beta6(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<double2> beta11(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();
  a5.H2D();
  a6.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();
  b5.H2D();
  b6.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();
  alpha5.H2D();
  alpha6.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();
  beta4.H2D();
  beta5.H2D();
  beta6.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha0.d_data, a0.d_data, CUDA_R_16F, 4, b0.d_data, CUDA_R_16F, 4, beta0.d_data, c0.d_data, CUDA_R_16F, 4, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha1.d_data, a1.d_data, CUDA_R_8I, 4, b1.d_data, CUDA_R_8I, 4, beta1.d_data, c1.d_data, CUDA_R_32I, 4, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha2.d_data, a2.d_data, CUDA_R_16BF, 4, b2.d_data, CUDA_R_16BF, 4, beta2.d_data, c2.d_data, CUDA_R_16BF, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha3.d_data, a3.d_data, CUDA_R_16F, 4, b3.d_data, CUDA_R_16F, 4, beta3.d_data, c3.d_data, CUDA_R_16F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha4.d_data, a4.d_data, CUDA_R_8I, 4, b4.d_data, CUDA_R_8I, 4, beta4.d_data, c4.d_data, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha5.d_data, a5.d_data, CUDA_R_16BF, 4, b5.d_data, CUDA_R_16BF, 4, beta5.d_data, c5.d_data, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha6.d_data, a6.d_data, CUDA_R_16F, 4, b6.d_data, CUDA_R_16F, 4, beta6.d_data, c6.d_data, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha7.d_data, a7.d_data, CUDA_R_32F, 4, b7.d_data, CUDA_R_32F, 4, beta7.d_data, c7.d_data, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha9.d_data, a9.d_data, CUDA_C_32F, 4, b9.d_data, CUDA_C_32F, 4, beta9.d_data, c9.d_data, CUDA_C_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha10.d_data, a10.d_data, CUDA_R_64F, 4, b10.d_data, CUDA_R_64F, 4, beta10.d_data, c10.d_data, CUDA_R_64F, 4, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha11.d_data, a11.d_data, CUDA_C_64F, 4, b11.d_data, CUDA_C_64F, 4, beta11.d_data, c11.d_data, CUDA_C_64F, 4, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();
  c4.D2H();
  c5.D2H();
  c6.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[16] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 16) &&
      compare_result(expect, c1.h_data, 16) &&
      compare_result(expect, c2.h_data, 16) &&
      compare_result(expect, c3.h_data, 16) &&
      compare_result(expect, c4.h_data, 16) &&
      compare_result(expect, c5.h_data, 16) &&
      compare_result(expect, c6.h_data, 16) &&
      compare_result(expect, c7.h_data, 16) &&
      compare_result(expect, c9.h_data, 16) &&
      compare_result(expect, c10.h_data, 16) &&
      compare_result(expect, c11.h_data, 16))
    printf("GemmEx pass\n");
  else {
    printf("GemmEx fail\n");
    test_passed = false;
  }
}

void test_cublasTsyrkx() {
  std::vector<float> v = {2, 3, 5, 7, 11, 13};
  Data<float> a0(v.data(), 6);
  Data<double> a1(v.data(), 6);
  Data<float2> a2(v.data(), 6);
  Data<double2> a3(v.data(), 6);

  Data<float> b0(v.data(), 6);
  Data<double> b1(v.data(), 6);
  Data<float2> b2(v.data(), 6);
  Data<double2> b3(v.data(), 6);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<float2> c2(4);
  Data<double2> c3(4);

  float alpha = 2;
  float beta = 0;
  Data<float> alpha0(&alpha, 1);
  Data<double> alpha1(&alpha, 1);
  Data<float2> alpha2(&alpha, 1);
  Data<double2> alpha3(&alpha, 1);

  Data<float> beta0(&beta, 1);
  Data<double> beta1(&beta, 1);
  Data<float2> beta2(&beta, 1);
  Data<double2> beta3(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();

  cublasSsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha0.d_data, a0.d_data, 3, b0.d_data, 3, beta0.d_data, c0.d_data, 2);
  cublasDsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha1.d_data, a1.d_data, 3, b1.d_data, 3, beta1.d_data, c1.d_data, 2);
  cublasCsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha2.d_data, a2.d_data, 3, b2.d_data, 3, beta2.d_data, c2.d_data, 2);
  cublasZsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha3.d_data, a3.d_data, 3, b3.d_data, 3, beta3.d_data, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = { 76.000000,0.000000,224.000000,678.000000 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Tsyrkx pass\n");
  else {
    printf("Tsyrkx fail\n");
    test_passed = false;
  }
}

void test_cublasTherkx() {
  std::vector<float> v = {2, 3, 5, 7, 11, 13};
  Data<float2> a0(v.data(), 6);
  Data<double2> a1(v.data(), 6);

  Data<float2> b0(v.data(), 6);
  Data<double2> b1(v.data(), 6);

  Data<float2> c0(4);
  Data<double2> c1(4);

  float alpha = 2;
  float beta = 0;
  Data<float2> alpha0(&alpha, 1);
  Data<double2> alpha1(&alpha, 1);

  Data<float> beta0(&beta, 1);
  Data<double> beta1(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();

  b0.H2D();
  b1.H2D();

  alpha0.H2D();
  alpha1.H2D();

  beta0.H2D();
  beta1.H2D();

  cublasCherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha0.d_data, a0.d_data, 3, b0.d_data, 3, beta0.d_data, c0.d_data, 2);
  cublasZherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha1.d_data, a1.d_data, 3, b1.d_data, 3, beta1.d_data, c1.d_data, 2);

  c0.D2H();
  c1.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = { 76.000000,0.000000,224.000000,678.000000 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4))
    printf("Therkx pass\n");
  else {
    printf("Therkx fail\n");
    test_passed = false;
  }
}

void test_cublasTdgmm() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> a0(v.data(), 4);
  Data<double> a1(v.data(), 4);
  Data<float2> a2(v.data(), 4);
  Data<double2> a3(v.data(), 4);

  std::vector<float> x = {2, 3};
  Data<float> x0(v.data(), 2);
  Data<double> x1(v.data(), 2);
  Data<float2> x2(v.data(), 2);
  Data<double2> x3(v.data(), 2);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<float2> c2(4);
  Data<double2> c3(4);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  x0.H2D();
  x1.H2D();
  x2.H2D();
  x3.H2D();

  cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a0.d_data, 2, x0.d_data, 1, c0.d_data, 2);
  cublasDdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a1.d_data, 2, x1.d_data, 1, c1.d_data, 2);
  cublasCdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a2.d_data, 2, x2.d_data, 1, c2.d_data, 2);
  cublasZdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a3.d_data, 2, x3.d_data, 1, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = { 4.0, 9.0, 10.0, 21.0 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Tdgmm pass\n");
  else {
    printf("Tdgmm fail\n");
    test_passed = false;
  }
}

struct Ptr_Data {
  int group_num;
  void** h_data;
  void** d_data;
  Ptr_Data(int group_num) : group_num(group_num) {
    h_data = (void**)malloc(group_num * sizeof(void*));
    memset(h_data, 0, group_num * sizeof(void*));
    cudaMalloc(&d_data, group_num * sizeof(void*));
    cudaMemset(d_data, 0, group_num * sizeof(void*));
  }
  ~Ptr_Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    cudaMemcpy(d_data, h_data, group_num * sizeof(void*), cudaMemcpyHostToDevice);
  }
};

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasGemmBatchedEx() {
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<half> a0(v.data(), 32);
  Data<std::int8_t> a1(v.data(), 32);
  Data<__nv_bfloat16> a2(v.data(), 32);
  Data<half> a3(v.data(), 32);
  Data<std::int8_t> a4(v.data(), 32);
  Data<__nv_bfloat16> a5(v.data(), 32);
  Data<half> a6(v.data(), 32);
  Data<float> a7(v.data(), 32);
  Data<float2> a9(v.data(), 32);
  Data<double> a10(v.data(), 32);
  Data<double2> a11(v.data(), 32);

  Ptr_Data a0_ptrs(2);  a0_ptrs.h_data[0] = a0.d_data; a0_ptrs.h_data[1] = a0.d_data + 16;
  Ptr_Data a1_ptrs(2);  a1_ptrs.h_data[0] = a1.d_data; a1_ptrs.h_data[1] = a1.d_data + 16;
  Ptr_Data a2_ptrs(2);  a2_ptrs.h_data[0] = a2.d_data; a2_ptrs.h_data[1] = a2.d_data + 16;
  Ptr_Data a3_ptrs(2);  a3_ptrs.h_data[0] = a3.d_data; a3_ptrs.h_data[1] = a3.d_data + 16;
  Ptr_Data a4_ptrs(2);  a4_ptrs.h_data[0] = a4.d_data; a4_ptrs.h_data[1] = a4.d_data + 16;
  Ptr_Data a5_ptrs(2);  a5_ptrs.h_data[0] = a5.d_data; a5_ptrs.h_data[1] = a5.d_data + 16;
  Ptr_Data a6_ptrs(2);  a6_ptrs.h_data[0] = a6.d_data; a6_ptrs.h_data[1] = a6.d_data + 16;
  Ptr_Data a7_ptrs(2);  a7_ptrs.h_data[0] = a7.d_data; a7_ptrs.h_data[1] = a7.d_data + 16;
  Ptr_Data a9_ptrs(2);  a9_ptrs.h_data[0] = a9.d_data; a9_ptrs.h_data[1] = a9.d_data + 16;
  Ptr_Data a10_ptrs(2); a10_ptrs.h_data[0] = a10.d_data; a10_ptrs.h_data[1] = a10.d_data + 16;
  Ptr_Data a11_ptrs(2); a11_ptrs.h_data[0] = a11.d_data; a11_ptrs.h_data[1] = a11.d_data + 16;

  Data<half> b0(v.data(), 32);
  Data<std::int8_t> b1(v.data(), 32);
  Data<__nv_bfloat16> b2(v.data(), 32);
  Data<half> b3(v.data(), 32);
  Data<std::int8_t> b4(v.data(), 32);
  Data<__nv_bfloat16> b5(v.data(), 32);
  Data<half> b6(v.data(), 32);
  Data<float> b7(v.data(), 32);
  Data<float2> b9(v.data(), 32);
  Data<double> b10(v.data(), 32);
  Data<double2> b11(v.data(), 32);

  Ptr_Data b0_ptrs(2);  b0_ptrs.h_data[0] = b0.d_data; b0_ptrs.h_data[1] = b0.d_data + 16;
  Ptr_Data b1_ptrs(2);  b1_ptrs.h_data[0] = b1.d_data; b1_ptrs.h_data[1] = b1.d_data + 16;
  Ptr_Data b2_ptrs(2);  b2_ptrs.h_data[0] = b2.d_data; b2_ptrs.h_data[1] = b2.d_data + 16;
  Ptr_Data b3_ptrs(2);  b3_ptrs.h_data[0] = b3.d_data; b3_ptrs.h_data[1] = b3.d_data + 16;
  Ptr_Data b4_ptrs(2);  b4_ptrs.h_data[0] = b4.d_data; b4_ptrs.h_data[1] = b4.d_data + 16;
  Ptr_Data b5_ptrs(2);  b5_ptrs.h_data[0] = b5.d_data; b5_ptrs.h_data[1] = b5.d_data + 16;
  Ptr_Data b6_ptrs(2);  b6_ptrs.h_data[0] = b6.d_data; b6_ptrs.h_data[1] = b6.d_data + 16;
  Ptr_Data b7_ptrs(2);  b7_ptrs.h_data[0] = b7.d_data; b7_ptrs.h_data[1] = b7.d_data + 16;
  Ptr_Data b9_ptrs(2);  b9_ptrs.h_data[0] = b9.d_data; b9_ptrs.h_data[1] = b9.d_data + 16;
  Ptr_Data b10_ptrs(2); b10_ptrs.h_data[0] = b10.d_data; b10_ptrs.h_data[1] = b10.d_data + 16;
  Ptr_Data b11_ptrs(2); b11_ptrs.h_data[0] = b11.d_data; b11_ptrs.h_data[1] = b11.d_data + 16;

  Data<half> c0(32);
  Data<std::int32_t> c1(32);
  Data<__nv_bfloat16> c2(32);
  Data<half> c3(32);
  Data<float> c4(32);
  Data<float> c5(32);
  Data<float> c6(32);
  Data<float> c7(32);
  Data<float2> c9(32);
  Data<double> c10(32);
  Data<double2> c11(32);

  Ptr_Data c0_ptrs(2);  c0_ptrs.h_data[0] = c0.d_data; c0_ptrs.h_data[1] = c0.d_data + 16;
  Ptr_Data c1_ptrs(2);  c1_ptrs.h_data[0] = c1.d_data; c1_ptrs.h_data[1] = c1.d_data + 16;
  Ptr_Data c2_ptrs(2);  c2_ptrs.h_data[0] = c2.d_data; c2_ptrs.h_data[1] = c2.d_data + 16;
  Ptr_Data c3_ptrs(2);  c3_ptrs.h_data[0] = c3.d_data; c3_ptrs.h_data[1] = c3.d_data + 16;
  Ptr_Data c4_ptrs(2);  c4_ptrs.h_data[0] = c4.d_data; c4_ptrs.h_data[1] = c4.d_data + 16;
  Ptr_Data c5_ptrs(2);  c5_ptrs.h_data[0] = c5.d_data; c5_ptrs.h_data[1] = c5.d_data + 16;
  Ptr_Data c6_ptrs(2);  c6_ptrs.h_data[0] = c6.d_data; c6_ptrs.h_data[1] = c6.d_data + 16;
  Ptr_Data c7_ptrs(2);  c7_ptrs.h_data[0] = c7.d_data; c7_ptrs.h_data[1] = c7.d_data + 16;
  Ptr_Data c9_ptrs(2);  c9_ptrs.h_data[0] = c9.d_data; c9_ptrs.h_data[1] = c9.d_data + 16;
  Ptr_Data c10_ptrs(2); c10_ptrs.h_data[0] = c10.d_data; c10_ptrs.h_data[1] = c10.d_data + 16;
  Ptr_Data c11_ptrs(2); c11_ptrs.h_data[0] = c11.d_data; c11_ptrs.h_data[1] = c11.d_data + 16; 

  float alpha = 2;
  float beta = 0;
  Data<half> alpha0(&alpha, 1);
  Data<std::int32_t> alpha1(&alpha, 1);
  Data<float> alpha2(&alpha, 1);
  Data<float> alpha3(&alpha, 1);
  Data<float> alpha4(&alpha, 1);
  Data<float> alpha5(&alpha, 1);
  Data<float> alpha6(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<double2> alpha11(&alpha, 1);

  Data<half> beta0(&beta, 1);
  Data<std::int32_t> beta1(&beta, 1);
  Data<float> beta2(&beta, 1);
  Data<float> beta3(&beta, 1);
  Data<float> beta4(&beta, 1);
  Data<float> beta5(&beta, 1);
  Data<float> beta6(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<double2> beta11(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();
  a5.H2D();
  a6.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();
  b5.H2D();
  b6.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  a0_ptrs.H2D();
  a1_ptrs.H2D();
  a2_ptrs.H2D();
  a3_ptrs.H2D();
  a4_ptrs.H2D();
  a5_ptrs.H2D();
  a6_ptrs.H2D();
  a7_ptrs.H2D();
  a9_ptrs.H2D();
  a10_ptrs.H2D();
  a11_ptrs.H2D();

  b0_ptrs.H2D();
  b1_ptrs.H2D();
  b2_ptrs.H2D();
  b3_ptrs.H2D();
  b4_ptrs.H2D();
  b5_ptrs.H2D();
  b6_ptrs.H2D();
  b7_ptrs.H2D();
  b9_ptrs.H2D();
  b10_ptrs.H2D();
  b11_ptrs.H2D();

  c0_ptrs.H2D();
  c1_ptrs.H2D();
  c2_ptrs.H2D();
  c3_ptrs.H2D();
  c4_ptrs.H2D();
  c5_ptrs.H2D();
  c6_ptrs.H2D();
  c7_ptrs.H2D();
  c9_ptrs.H2D();
  c10_ptrs.H2D();
  c11_ptrs.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();
  alpha5.H2D();
  alpha6.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();
  beta4.H2D();
  beta5.H2D();
  beta6.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha0.d_data, (const void**)a0_ptrs.d_data, CUDA_R_16F, 4, (const void**)b0_ptrs.d_data, CUDA_R_16F, 4, beta0.d_data, c0_ptrs.d_data, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha1.d_data, (const void**)a1_ptrs.d_data, CUDA_R_8I, 4, (const void**)b1_ptrs.d_data, CUDA_R_8I, 4, beta1.d_data, c1_ptrs.d_data, CUDA_R_32I, 4, 2, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha2.d_data, (const void**)a2_ptrs.d_data, CUDA_R_16BF, 4, (const void**)b2_ptrs.d_data, CUDA_R_16BF, 4, beta2.d_data, c2_ptrs.d_data, CUDA_R_16BF, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha3.d_data, (const void**)a3_ptrs.d_data, CUDA_R_16F, 4, (const void**)b3_ptrs.d_data, CUDA_R_16F, 4, beta3.d_data, c3_ptrs.d_data, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha4.d_data, (const void**)a4_ptrs.d_data, CUDA_R_8I, 4, (const void**)b4_ptrs.d_data, CUDA_R_8I, 4, beta4.d_data, c4_ptrs.d_data, CUDA_R_32F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha5.d_data, (const void**)a5_ptrs.d_data, CUDA_R_16BF, 4, (const void**)b5_ptrs.d_data, CUDA_R_16BF, 4, beta5.d_data, c5_ptrs.d_data, CUDA_R_32F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha6.d_data, (const void**)a6_ptrs.d_data, CUDA_R_16F, 4, (const void**)b6_ptrs.d_data, CUDA_R_16F, 4, beta6.d_data, c6_ptrs.d_data, CUDA_R_32F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha7.d_data, (const void**)a7_ptrs.d_data, CUDA_R_32F, 4, (const void**)b7_ptrs.d_data, CUDA_R_32F, 4, beta7.d_data, c7_ptrs.d_data, CUDA_R_32F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha9.d_data, (const void**)a9_ptrs.d_data, CUDA_C_32F, 4, (const void**)b9_ptrs.d_data, CUDA_C_32F, 4, beta9.d_data, c9_ptrs.d_data, CUDA_C_32F, 4, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha10.d_data, (const void**)a10_ptrs.d_data, CUDA_R_64F, 4, (const void**)b10_ptrs.d_data, CUDA_R_64F, 4, beta10.d_data, c10_ptrs.d_data, CUDA_R_64F, 4, 2, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha11.d_data, (const void**)a11_ptrs.d_data, CUDA_C_64F, 4, (const void**)b11_ptrs.d_data, CUDA_C_64F, 4, beta11.d_data, c11_ptrs.d_data, CUDA_C_64F, 4, 2, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();
  c4.D2H();
  c5.D2H();
  c6.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c1.h_data, 32) &&
      compare_result(expect, c2.h_data, 32) &&
      compare_result(expect, c3.h_data, 32) &&
      compare_result(expect, c4.h_data, 32) &&
      compare_result(expect, c5.h_data, 32) &&
      compare_result(expect, c6.h_data, 32) &&
      compare_result(expect, c7.h_data, 32) &&
      compare_result(expect, c9.h_data, 32) &&
      compare_result(expect, c10.h_data, 32) &&
      compare_result(expect, c11.h_data, 32))
    printf("GemmBatchedEx pass\n");
  else {
    printf("GemmBatchedEx fail\n");
    test_passed = false;
  }
}
#endif

void test_cublasGemmStridedBatchedEx() {
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  std::vector<float> v2 = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                            2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<half> a0(v.data(), 32);
  Data<std::int8_t> a1(v.data(), 32);
  Data<__nv_bfloat16> a2(v.data(), 32);
  Data<half> a3(v.data(), 32);
  Data<std::int8_t> a4(v.data(), 32);
  Data<__nv_bfloat16> a5(v.data(), 32);
  Data<half> a6(v.data(), 32);
  Data<float> a7(v.data(), 32);
  Data<float2> a9(v.data(), 32);
  Data<double> a10(v.data(), 32);
  Data<double2> a11(v.data(), 32);

  Data<half> b0(v.data(), 32);
  Data<std::int8_t> b1(v.data(), 32);
  Data<__nv_bfloat16> b2(v.data(), 32);
  Data<half> b3(v.data(), 32);
  Data<std::int8_t> b4(v.data(), 32);
  Data<__nv_bfloat16> b5(v.data(), 32);
  Data<half> b6(v.data(), 32);
  Data<float> b7(v.data(), 32);
  Data<float2> b9(v.data(), 32);
  Data<double> b10(v.data(), 32);
  Data<double2> b11(v.data(), 32);

  Data<half> c0(32);
  Data<std::int32_t> c1(32);
  Data<__nv_bfloat16> c2(32);
  Data<half> c3(32);
  Data<float> c4(32);
  Data<float> c5(32);
  Data<float> c6(32);
  Data<float> c7(32);
  Data<float2> c9(32);
  Data<double> c10(32);
  Data<double2> c11(32);

  float alpha = 2;
  float beta = 0;
  Data<half> alpha0(&alpha, 1);
  Data<std::int32_t> alpha1(&alpha, 1);
  Data<float> alpha2(&alpha, 1);
  Data<float> alpha3(&alpha, 1);
  Data<float> alpha4(&alpha, 1);
  Data<float> alpha5(&alpha, 1);
  Data<float> alpha6(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<double2> alpha11(&alpha, 1);

  Data<half> beta0(&beta, 1);
  Data<std::int32_t> beta1(&beta, 1);
  Data<float> beta2(&beta, 1);
  Data<float> beta3(&beta, 1);
  Data<float> beta4(&beta, 1);
  Data<float> beta5(&beta, 1);
  Data<float> beta6(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<double2> beta11(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();
  a5.H2D();
  a6.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();
  b5.H2D();
  b6.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();
  alpha5.H2D();
  alpha6.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();
  beta4.H2D();
  beta5.H2D();
  beta6.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha0.d_data, a0.d_data, CUDA_R_16F, 4, 16, b0.d_data, CUDA_R_16F, 4, 16, beta0.d_data, c0.d_data, CUDA_R_16F, 4, 16, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha1.d_data, a1.d_data, CUDA_R_8I, 4, 16, b1.d_data, CUDA_R_8I, 4, 16, beta1.d_data, c1.d_data, CUDA_R_32I, 4, 16, 2, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha2.d_data, a2.d_data, CUDA_R_16BF, 4, 16, b2.d_data, CUDA_R_16BF, 4, 16, beta2.d_data, c2.d_data, CUDA_R_16BF, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha3.d_data, a3.d_data, CUDA_R_16F, 4, 16, b3.d_data, CUDA_R_16F, 4, 16, beta3.d_data, c3.d_data, CUDA_R_16F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha4.d_data, a4.d_data, CUDA_R_8I, 4, 16, b4.d_data, CUDA_R_8I, 4, 16, beta4.d_data, c4.d_data, CUDA_R_32F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha5.d_data, a5.d_data, CUDA_R_16BF, 4, 16, b5.d_data, CUDA_R_16BF, 4, 16, beta5.d_data, c5.d_data, CUDA_R_32F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha6.d_data, a6.d_data, CUDA_R_16F, 4, 16, b6.d_data, CUDA_R_16F, 4, 16, beta6.d_data, c6.d_data, CUDA_R_32F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha7.d_data, a7.d_data, CUDA_R_32F, 4, 16, b7.d_data, CUDA_R_32F, 4, 16, beta7.d_data, c7.d_data, CUDA_R_32F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha9.d_data, a9.d_data, CUDA_C_32F, 4, 16, b9.d_data, CUDA_C_32F, 4, 16, beta9.d_data, c9.d_data, CUDA_C_32F, 4, 16, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha10.d_data, a10.d_data, CUDA_R_64F, 4, 16, b10.d_data, CUDA_R_64F, 4, 16, beta10.d_data, c10.d_data, CUDA_R_64F, 4, 16, 2, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha11.d_data, a11.d_data, CUDA_C_64F, 4, 16, b11.d_data, CUDA_C_64F, 4, 16, beta11.d_data, c11.d_data, CUDA_C_64F, 4, 16, 2, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();
  c4.D2H();
  c5.D2H();
  c6.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c1.h_data, 32) &&
      compare_result(expect, c2.h_data, 32) &&
      compare_result(expect, c3.h_data, 32) &&
      compare_result(expect, c4.h_data, 32) &&
      compare_result(expect, c5.h_data, 32) &&
      compare_result(expect, c6.h_data, 32) &&
      compare_result(expect, c7.h_data, 32) &&
      compare_result(expect, c9.h_data, 32) &&
      compare_result(expect, c10.h_data, 32) &&
      compare_result(expect, c11.h_data, 32))
    printf("GemmStridedBatchedEx pass\n");
  else {
    printf("GemmStridedBatchedEx fail\n");
    test_passed = false;
  }
}

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasTgemmBatched() {
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<half> a0(v.data(), 32);
  Data<float> a1(v.data(), 32);
  Data<float2> a2(v.data(), 32);
  Data<double> a3(v.data(), 32);
  Data<double2> a4(v.data(), 32);

  Ptr_Data a0_ptrs(2);  a0_ptrs.h_data[0] = a0.d_data; a0_ptrs.h_data[1] = a0.d_data + 16;
  Ptr_Data a1_ptrs(2);  a1_ptrs.h_data[0] = a1.d_data; a1_ptrs.h_data[1] = a1.d_data + 16;
  Ptr_Data a2_ptrs(2);  a2_ptrs.h_data[0] = a2.d_data; a2_ptrs.h_data[1] = a2.d_data + 16;
  Ptr_Data a3_ptrs(2); a3_ptrs.h_data[0] = a3.d_data; a3_ptrs.h_data[1] = a3.d_data + 16;
  Ptr_Data a4_ptrs(2); a4_ptrs.h_data[0] = a4.d_data; a4_ptrs.h_data[1] = a4.d_data + 16;

  Data<half> b0(v.data(), 32);
  Data<float> b1(v.data(), 32);
  Data<float2> b2(v.data(), 32);
  Data<double> b3(v.data(), 32);
  Data<double2> b4(v.data(), 32);

  Ptr_Data b0_ptrs(2);  b0_ptrs.h_data[0] = b0.d_data; b0_ptrs.h_data[1] = b0.d_data + 16;
  Ptr_Data b1_ptrs(2);  b1_ptrs.h_data[0] = b1.d_data; b1_ptrs.h_data[1] = b1.d_data + 16;
  Ptr_Data b2_ptrs(2);  b2_ptrs.h_data[0] = b2.d_data; b2_ptrs.h_data[1] = b2.d_data + 16;
  Ptr_Data b3_ptrs(2); b3_ptrs.h_data[0] = b3.d_data; b3_ptrs.h_data[1] = b3.d_data + 16;
  Ptr_Data b4_ptrs(2); b4_ptrs.h_data[0] = b4.d_data; b4_ptrs.h_data[1] = b4.d_data + 16;

  Data<half> c0(32);
  Data<float> c1(32);
  Data<float2> c2(32);
  Data<double> c3(32);
  Data<double2> c4(32);

  Ptr_Data c0_ptrs(2);  c0_ptrs.h_data[0] = c0.d_data; c0_ptrs.h_data[1] = c0.d_data + 16;
  Ptr_Data c1_ptrs(2);  c1_ptrs.h_data[0] = c1.d_data; c1_ptrs.h_data[1] = c1.d_data + 16;
  Ptr_Data c2_ptrs(2);  c2_ptrs.h_data[0] = c2.d_data; c2_ptrs.h_data[1] = c2.d_data + 16;
  Ptr_Data c3_ptrs(2); c3_ptrs.h_data[0] = c3.d_data; c3_ptrs.h_data[1] = c3.d_data + 16;
  Ptr_Data c4_ptrs(2); c4_ptrs.h_data[0] = c4.d_data; c4_ptrs.h_data[1] = c4.d_data + 16;

  float alpha = 2;
  float beta = 0;
  Data<half> alpha0(&alpha, 1);
  Data<float> alpha1(&alpha, 1);
  Data<float2> alpha2(&alpha, 1);
  Data<double> alpha3(&alpha, 1);
  Data<double2> alpha4(&alpha, 1);

  Data<half> beta0(&beta, 1);
  Data<float> beta1(&beta, 1);
  Data<float2> beta2(&beta, 1);
  Data<double> beta3(&beta, 1);
  Data<double2> beta4(&beta, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();

  a0_ptrs.H2D();
  a1_ptrs.H2D();
  a2_ptrs.H2D();
  a3_ptrs.H2D();
  a4_ptrs.H2D();

  b0_ptrs.H2D();
  b1_ptrs.H2D();
  b2_ptrs.H2D();
  b3_ptrs.H2D();
  b4_ptrs.H2D();

  c0_ptrs.H2D();
  c1_ptrs.H2D();
  c2_ptrs.H2D();
  c3_ptrs.H2D();
  c4_ptrs.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();
  beta4.H2D();

  cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha0.d_data, (const half**)a0_ptrs.d_data, 4, (const half**)b0_ptrs.d_data, 4, beta0.d_data, (half**)c0_ptrs.d_data, 4, 2);
  cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha1.d_data, (const float**)a1_ptrs.d_data, 4, (const float**)b1_ptrs.d_data, 4, beta1.d_data, (float**)c1_ptrs.d_data, 4, 2);
  cublasCgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha2.d_data, (const float2**)a2_ptrs.d_data, 4, (const float2**)b2_ptrs.d_data, 4, beta2.d_data, (float2**)c2_ptrs.d_data, 4, 2);
  cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha3.d_data, (const double**)a3_ptrs.d_data, 4, (const double**)b3_ptrs.d_data, 4, beta3.d_data, (double**)c3_ptrs.d_data, 4, 2);
  cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha4.d_data, (const double2**)a4_ptrs.d_data, 4, (const double2**)b4_ptrs.d_data, 4, beta4.d_data, (double2**)c4_ptrs.d_data, 4, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();
  c4.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c1.h_data, 32) &&
      compare_result(expect, c2.h_data, 32) &&
      compare_result(expect, c3.h_data, 32) &&
      compare_result(expect, c4.h_data, 32))
    printf("TgemmBatched pass\n");
  else {
    printf("TgemmBatched fail\n");
    test_passed = false;
  }
}
#endif

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasTtrsmBatched() {
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<float> a1(v.data(), 32);
  Data<float2> a2(v.data(), 32);
  Data<double> a3(v.data(), 32);
  Data<double2> a4(v.data(), 32);

  Ptr_Data a1_ptrs(2);  a1_ptrs.h_data[0] = a1.d_data; a1_ptrs.h_data[1] = a1.d_data + 16;
  Ptr_Data a2_ptrs(2);  a2_ptrs.h_data[0] = a2.d_data; a2_ptrs.h_data[1] = a2.d_data + 16;
  Ptr_Data a3_ptrs(2); a3_ptrs.h_data[0] = a3.d_data; a3_ptrs.h_data[1] = a3.d_data + 16;
  Ptr_Data a4_ptrs(2); a4_ptrs.h_data[0] = a4.d_data; a4_ptrs.h_data[1] = a4.d_data + 16;

  Data<float> b1(v.data(), 32);
  Data<float2> b2(v.data(), 32);
  Data<double> b3(v.data(), 32);
  Data<double2> b4(v.data(), 32);

  Ptr_Data b1_ptrs(2);  b1_ptrs.h_data[0] = b1.d_data; b1_ptrs.h_data[1] = b1.d_data + 16;
  Ptr_Data b2_ptrs(2);  b2_ptrs.h_data[0] = b2.d_data; b2_ptrs.h_data[1] = b2.d_data + 16;
  Ptr_Data b3_ptrs(2); b3_ptrs.h_data[0] = b3.d_data; b3_ptrs.h_data[1] = b3.d_data + 16;
  Ptr_Data b4_ptrs(2); b4_ptrs.h_data[0] = b4.d_data; b4_ptrs.h_data[1] = b4.d_data + 16;

  float alpha = 2;
  Data<float> alpha1(&alpha, 1);
  Data<float2> alpha2(&alpha, 1);
  Data<double> alpha3(&alpha, 1);
  Data<double2> alpha4(&alpha, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();

  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();

  a1_ptrs.H2D();
  a2_ptrs.H2D();
  a3_ptrs.H2D();
  a4_ptrs.H2D();

  b1_ptrs.H2D();
  b2_ptrs.H2D();
  b3_ptrs.H2D();
  b4_ptrs.H2D();

  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();

  cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha1.d_data, (const float**)a1_ptrs.d_data, 4, (float**)b1_ptrs.d_data, 4, 2);
  cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha2.d_data, (const float2**)a2_ptrs.d_data, 4, (float2**)b2_ptrs.d_data, 4, 2);
  cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha3.d_data, (const double**)a3_ptrs.d_data, 4, (double**)b3_ptrs.d_data, 4, 2);
  cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha4.d_data, (const double2**)a4_ptrs.d_data, 4, (double2**)b4_ptrs.d_data, 4, 2);

  b1.D2H();
  b2.D2H();
  b3.D2H();
  b4.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[32] = { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,
                       0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
  if (compare_result(expect, b1.h_data, 32) &&
      compare_result(expect, b2.h_data, 32) &&
      compare_result(expect, b3.h_data, 32) &&
      compare_result(expect, b4.h_data, 32))
    printf("TtrsmBatched pass\n");
  else {
    printf("TtrsmBatched fail\n");
    test_passed = false;
  }
}
#endif

void test_cublasTtrmm() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> a0(v.data(), 4);
  Data<double> a1(v.data(), 4);
  Data<float2> a2(v.data(), 4);
  Data<double2> a3(v.data(), 4);

  Data<float> b0(v.data(), 4);
  Data<double> b1(v.data(), 4);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<float2> c2(v.data(), 4);
  Data<double2> c3(v.data(), 4);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  b0.H2D();
  b1.H2D();

  c2.H2D();
  c3.H2D();

  float alpha = 2;
  Data<float> alpha0(&alpha, 1);
  Data<double> alpha1(&alpha, 1);
  Data<float2> alpha2(&alpha, 1);
  Data<double2> alpha3(&alpha, 1);

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();

  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, alpha0.d_data, a0.d_data, 2, b0.d_data, 2, c0.d_data, 2);
  cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, alpha1.d_data, a1.d_data, 2, b1.d_data, 2, c1.d_data, 2);
  cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, alpha2.d_data, a2.d_data, 2, c2.d_data, 2, c2.d_data, 2);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, alpha3.d_data, a3.d_data, 2, c3.d_data, 2, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  cudaStreamSynchronize(0);

  cublasDestroy(handle);

  float expect[4] = { 38.0, 42.0, 90.0, 98.0 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Ttrmm pass\n");
  else {
    printf("Ttrmm fail\n");
    test_passed = false;
  }
}

void test_cublasTrot() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float2> x0(v.data(), 4);
  Data<double2> x1(v.data(), 4);
  Data<float2> y0(v.data(), 4);
  Data<double2> y1(v.data(), 4);

  float cos = 0.866;
  float sin = 0.5;
  Data<float> cos0(&cos, 1);
  Data<double> cos1(&cos, 1);
  Data<float2> sin0(&sin, 1);
  Data<double2> sin1(&sin, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x0.H2D();
  x1.H2D();
  y0.H2D();
  y1.H2D();
  sin0.H2D();
  sin1.H2D();
  cos0.H2D();
  cos1.H2D();

  cublasCrot(handle, 4, x0.d_data, 1,  y0.d_data, 1, cos0.d_data, sin0.d_data);
  cublasZrot(handle, 4, x1.d_data, 1,  y1.d_data, 1, cos1.d_data, sin1.d_data);

  x0.D2H();
  x1.D2H();
  y0.D2H();
  y1.D2H();

  cudaStreamSynchronize(0);
  cublasDestroy(handle);

  float expect_x[4] = {2.732000,4.098000,6.830000,9.562000};
  float expect_y[4] = {0.732000,1.098000,1.830000,2.562000};
  if (compare_result(expect_x, x0.h_data, 4) &&
      compare_result(expect_x, x1.h_data, 4) &&
      compare_result(expect_y, y0.h_data, 4) &&
      compare_result(expect_y, y1.h_data, 4))
    printf("Trot pass\n");
  else {
    printf("Trot fail\n");
    test_passed = false;
  }
}

void test_cublasTsymv() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float2> a0(v.data(), 4);
  Data<double2> a1(v.data(), 4);
  Data<float2> x0(v.data(), 2);
  Data<double2> x1(v.data(), 2);
  Data<float2> y0(2);
  Data<double2> y1(2);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x0.H2D();
  x1.H2D();
  a0.H2D();
  a1.H2D();

  float alpha = 2;
  float beta = 3;
  Data<float2> alpha0(&alpha, 1);
  Data<double2> alpha1(&alpha, 1);
  alpha0.H2D();
  alpha1.H2D();
  Data<float2> beta0(&beta, 1);
  Data<double2> beta1(&beta, 1);
  beta0.H2D();
  beta1.H2D();

  cublasCsymv(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha0.d_data, a0.d_data, 2, x0.d_data, 1, beta0.d_data, y0.d_data, 1);
  cublasZsymv(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha1.d_data, a1.d_data, 2, x1.d_data, 1, beta1.d_data, y1.d_data, 1);

  y0.D2H();
  y1.D2H();

  cudaStreamSynchronize(0);
  cublasDestroy(handle);

  float expect_y[2] = {38.000000, 62.000000};
  if (compare_result(expect_y, y0.h_data, 2) &&
      compare_result(expect_y, y1.h_data, 2))
    printf("Tsymv pass\n");
  else {
    printf("Tsymv fail\n");
    test_passed = false;
  }
}

void test_cublasTsyr() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float2> a0(v.data(), 4);
  Data<double2> a1(v.data(), 4);
  Data<float2> x0(v.data(), 2);
  Data<double2> x1(v.data(), 2);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x0.H2D();
  x1.H2D();
  a0.H2D();
  a1.H2D();

  float alpha = 2;
  Data<float2> alpha0(&alpha, 1);
  Data<double2> alpha1(&alpha, 1);
  alpha0.H2D();
  alpha1.H2D();

  cublasCsyr(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha0.d_data, x0.d_data, 1, a0.d_data, 2);
  cublasZsyr(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha1.d_data, x1.d_data, 1, a1.d_data, 2);

  a0.D2H();
  a1.D2H();

  cudaStreamSynchronize(0);
  cublasDestroy(handle);

  float expect_a[4] = {10.000000, 3.000000, 17.000000, 25.000000};
  if (compare_result(expect_a, a0.h_data, 4) &&
      compare_result(expect_a, a1.h_data, 4))
    printf("Tsyr pass\n");
  else {
    printf("Tsyr fail\n");
    test_passed = false;
  }
}

void test_cublasTsyr2() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float2> a0(v.data(), 4);
  Data<double2> a1(v.data(), 4);
  Data<float2> x0(v.data(), 2);
  Data<double2> x1(v.data(), 2);
  Data<float2> y0(v.data(), 2);
  Data<double2> y1(v.data(), 2);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  x0.H2D();
  x1.H2D();
  y0.H2D();
  y1.H2D();
  a0.H2D();
  a1.H2D();

  float alpha = 2;
  Data<float2> alpha0(&alpha, 1);
  Data<double2> alpha1(&alpha, 1);
  alpha0.H2D();
  alpha1.H2D();

  cublasCsyr2(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha0.d_data, x0.d_data, 1, y0.d_data, 1, a0.d_data, 2);
  cublasZsyr2(handle, CUBLAS_FILL_MODE_UPPER, 2, alpha1.d_data, x1.d_data, 1, y1.d_data, 1, a1.d_data, 2);

  a0.D2H();
  a1.D2H();

  cudaStreamSynchronize(0);
  cublasDestroy(handle);

  float expect_a[4] = {18.000000, 3.000000, 29.000000, 43.000000};
  if (compare_result(expect_a, a0.h_data, 4) &&
      compare_result(expect_a, a1.h_data, 4))
    printf("Tsyr2 pass\n");
  else {
    printf("Tsyr2 fail\n");
    test_passed = false;
  }
}

void test_cublasTgeam() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> a0(v.data(), 4);
  Data<double> a1(v.data(), 4);
  Data<float2> a2(v.data(), 4);
  Data<double2> a3(v.data(), 4);
  Data<float> b0(v.data(), 4);
  Data<double> b1(v.data(), 4);
  Data<float2> b2(v.data(), 4);
  Data<double2> b3(v.data(), 4);
  Data<float> c0(4);
  Data<double> c1(4);
  Data<float2> c2(4);
  Data<double2> c3(4);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();

  float alpha = 2;
  float beta = 3;
  Data<float> alpha0(&alpha, 1);
  Data<double> alpha1(&alpha, 1);
  Data<float2> alpha2(&alpha, 1);
  Data<double2> alpha3(&alpha, 1);
  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  Data<float> beta0(&beta, 1);
  Data<double> beta1(&beta, 1);
  Data<float2> beta2(&beta, 1);
  Data<double2> beta3(&beta, 1);
  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();

  cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, alpha0.d_data, a0.d_data, 2, beta0.d_data, b0.d_data, 2, c0.d_data, 2);
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, alpha1.d_data, a1.d_data, 2, beta1.d_data, b1.d_data, 2, c1.d_data, 2);
  cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, alpha2.d_data, a2.d_data, 2, beta2.d_data, b2.d_data, 2, c2.d_data, 2);
  cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, alpha3.d_data, a3.d_data, 2, beta3.d_data, b3.d_data, 2, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  cudaStreamSynchronize(0);
  cublasDestroy(handle);

  float expect_c[4] = {10.000000, 15.000000, 25.000000, 35.000000};
  if (compare_result(expect_c, c0.h_data, 4) &&
      compare_result(expect_c, c1.h_data, 4) &&
      compare_result(expect_c, c2.h_data, 4) &&
      compare_result(expect_c, c3.h_data, 4))
    printf("Tgeam pass\n");
  else {
    printf("Tgeam fail\n");
    test_passed = false;
  }
}

int main() {
  test_cublasNrm2Ex();
  test_cublasDotEx();
  test_cublasDotcEx();
  test_cublasScalEx();
  test_cublasAxpyEx();
  test_cublasRotEx();
  test_cublasGemmEx();
  test_cublasTsyrkx();
  test_cublasTherkx();
  test_cublasTdgmm();
#ifndef DPCT_USM_LEVEL_NONE
  test_cublasGemmBatchedEx();
#endif
  test_cublasGemmStridedBatchedEx();
#ifndef DPCT_USM_LEVEL_NONE
  test_cublasTgemmBatched();
  test_cublasTtrsmBatched();
#endif
  test_cublasTtrmm();
  test_cublasTrot();
  test_cublasTsymv();
  test_cublasTsyr();
  test_cublasTsyr2();
  test_cublasTgeam();

  if (test_passed)
    return 0;
  return -1;
}

