// ===------ cusolver_test3.cu ------------------------------*- CUDA -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusolverDn.h"

#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>

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
inline void Data<float2>::to_float_convert(float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool compare_result(float* expect, float* result, std::vector<int> indices) {
  for (int i = 0; i < indices.size(); i++) {
    if (std::abs(result[indices[i]]-expect[indices[i]]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cusolverDnXgetrf() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(2);
  Data<int64_t> ipiv_d(2);
  Data<int64_t> ipiv_c(2);
  Data<int64_t> ipiv_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);
  cudaMalloc(&host_ws_s, host_ws_size_s);
  cudaMalloc(&host_ws_d, host_ws_size_d);
  cudaMalloc(&host_ws_c, host_ws_size_c);
  cudaMalloc(&host_ws_z, host_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, ipiv_s.d_data, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, ipiv_d.d_data, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, ipiv_c.d_data, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, ipiv_z.d_data, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  ipiv_s.D2H();
  ipiv_d.D2H();
  ipiv_c.D2H();
  ipiv_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(host_ws_s);
  cudaFree(host_ws_d);
  cudaFree(host_ws_c);
  cudaFree(host_ws_z);
  cudaFree(info);

  float expect_a[4] = {2, 0.5, 4, 1};
  float expect_ipiv[2] = {2, 2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_ipiv, ipiv_s.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_d.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_c.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_z.h_data, 2))
    printf("DnXgetrf pass\n");
  else {
    printf("DnXgetrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgetrfnp() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);
  cudaMalloc(&host_ws_s, host_ws_size_s);
  cudaMalloc(&host_ws_d, host_ws_size_d);
  cudaMalloc(&host_ws_c, host_ws_size_c);
  cudaMalloc(&host_ws_z, host_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, nullptr, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, nullptr, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, nullptr, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, nullptr, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(host_ws_s);
  cudaFree(host_ws_d);
  cudaFree(host_ws_c);
  cudaFree(host_ws_z);
  cudaFree(info);

  float expect_a[4] = {1, 2, 3, -2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4))
    printf("DnXgetrfnp pass\n");
  else {
    printf("DnXgetrfnp fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGetrf() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(2);
  Data<int64_t> ipiv_d(2);
  Data<int64_t> ipiv_c(2);
  Data<int64_t> ipiv_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, &device_ws_size_s);
  cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, &device_ws_size_d);
  cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, &device_ws_size_c);
  cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;

  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnGetrf(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, ipiv_s.d_data, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnGetrf(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, ipiv_d.d_data, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnGetrf(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, ipiv_c.d_data, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnGetrf(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, ipiv_z.d_data, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  ipiv_s.D2H();
  ipiv_d.D2H();
  ipiv_c.D2H();
  ipiv_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {2, 0.5, 4, 1};
  float expect_ipiv[2] = {2, 2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_ipiv, ipiv_s.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_d.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_c.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_z.h_data, 2))
    printf("DnGetrf pass\n");
  else {
    printf("DnGetrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgetrs() {
  std::vector<float> a = {2, 0.5, 4, 1};
  std::vector<float> ipiv = {2, 2};
  std::vector<float> b = {23, 34, 31, 46, 39, 58};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(ipiv.data(), 2);
  Data<int64_t> ipiv_d(ipiv.data(), 2);
  Data<int64_t> ipiv_c(ipiv.data(), 2);
  Data<int64_t> ipiv_z(ipiv.data(), 2);
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<float2> b_c(b.data(), 6);
  Data<double2> b_z(b.data(), 6);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s.d_data, 2, ipiv_s.d_data, CUDA_R_32F, b_s.d_data, 2, info);
  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_64F, a_d.d_data, 2, ipiv_d.d_data, CUDA_R_64F, b_d.d_data, 2, info);
  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_32F, a_c.d_data, 2, ipiv_c.d_data, CUDA_C_32F, b_c.d_data, 2, info);
  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_64F, a_z.d_data, 2, ipiv_z.d_data, CUDA_C_64F, b_z.d_data, 2, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(info);

  float expect_b[6] = {5, 6, 7, 8, 9, 10};
  if (compare_result(expect_b, b_s.h_data, 6) &&
      compare_result(expect_b, b_d.h_data, 6) &&
      compare_result(expect_b, b_c.h_data, 6) &&
      compare_result(expect_b, b_z.h_data, 6))
    printf("DnXgetrs pass\n");
  else {
    printf("DnXgetrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGetrs() {
  std::vector<float> a = {2, 0.5, 4, 1};
  std::vector<float> ipiv = {2, 2};
  std::vector<float> b = {23, 34, 31, 46, 39, 58};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(ipiv.data(), 2);
  Data<int64_t> ipiv_d(ipiv.data(), 2);
  Data<int64_t> ipiv_c(ipiv.data(), 2);
  Data<int64_t> ipiv_z(ipiv.data(), 2);
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<float2> b_c(b.data(), 6);
  Data<double2> b_z(b.data(), 6);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s.d_data, 2, ipiv_s.d_data, CUDA_R_32F, b_s.d_data, 2, info);
  cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_64F, a_d.d_data, 2, ipiv_d.d_data, CUDA_R_64F, b_d.d_data, 2, info);
  cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_32F, a_c.d_data, 2, ipiv_c.d_data, CUDA_C_32F, b_c.d_data, 2, info);
  cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_64F, a_z.d_data, 2, ipiv_z.d_data, CUDA_C_64F, b_z.d_data, 2, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(info);

  float expect_b[6] = {5, 6, 7, 8, 9, 10};
  if (compare_result(expect_b, b_s.h_data, 6) &&
      compare_result(expect_b, b_d.h_data, 6) &&
      compare_result(expect_b, b_c.h_data, 6) &&
      compare_result(expect_b, b_z.h_data, 6))
    printf("DnGetrs pass\n");
  else {
    printf("DnGetrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgeqrf() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> tau_s(2);
  Data<double> tau_d(2);
  Data<float2> tau_c(2);
  Data<double2> tau_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  tau_s.H2D();
  tau_d.H2D();
  tau_c.H2D();
  tau_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, tau_s.d_data, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, tau_d.d_data, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, tau_c.d_data, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, tau_z.d_data, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);
  cudaMalloc(&host_ws_s, host_ws_size_s);
  cudaMalloc(&host_ws_d, host_ws_size_d);
  cudaMalloc(&host_ws_c, host_ws_size_c);
  cudaMalloc(&host_ws_z, host_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXgeqrf(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, tau_s.d_data, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXgeqrf(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, tau_d.d_data, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXgeqrf(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, tau_c.d_data, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXgeqrf(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, tau_z.d_data, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  tau_s.D2H();
  tau_d.D2H();
  tau_c.D2H();
  tau_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(host_ws_s);
  cudaFree(host_ws_d);
  cudaFree(host_ws_c);
  cudaFree(host_ws_z);
  cudaFree(info);

  float expect_a[4] = {-2.236068, 0.618034, -4.919349, -0.894427};
  float expect_tau[2] = {1.447214, 0};

  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_tau, tau_s.h_data, 2) &&
      compare_result(expect_tau, tau_d.h_data, 2) &&
      compare_result(expect_tau, tau_c.h_data, 2) &&
      compare_result(expect_tau, tau_z.h_data, 2))
    printf("DnXgeqrf pass\n");
  else {
    printf("DnXgeqrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGeqrf() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> tau_s(2);
  Data<double> tau_d(2);
  Data<float2> tau_c(2);
  Data<double2> tau_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  tau_s.H2D();
  tau_d.H2D();
  tau_c.H2D();
  tau_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, tau_s.d_data, CUDA_R_32F, &device_ws_size_s);
  cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, tau_d.d_data, CUDA_R_64F, &device_ws_size_d);
  cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, tau_c.d_data, CUDA_C_32F, &device_ws_size_c);
  cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, tau_z.d_data, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnGeqrf(handle, params, 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, tau_s.d_data, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnGeqrf(handle, params, 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, tau_d.d_data, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnGeqrf(handle, params, 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_C_32F, tau_c.d_data, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnGeqrf(handle, params, 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_C_64F, tau_z.d_data, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  tau_s.D2H();
  tau_d.D2H();
  tau_c.D2H();
  tau_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {-2.236068, 0.618034, -4.919349, -0.894427};
  float expect_tau[2] = {1.447214, 0};

  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_tau, tau_s.h_data, 2) &&
      compare_result(expect_tau, tau_d.h_data, 2) &&
      compare_result(expect_tau, tau_c.h_data, 2) &&
      compare_result(expect_tau, tau_z.h_data, 2))
    printf("DnGeqrf pass\n");
  else {
    printf("DnGeqrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgesvd() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);

  Data<float> s_s(2);
  Data<double> s_d(2);
  Data<float> s_c(2);
  Data<double> s_z(2);

  Data<float> u_s(4);
  Data<double> u_d(4);
  Data<float2> u_c(4);
  Data<double2> u_z(4);

  Data<float> vt_s(4);
  Data<double> vt_d(4);
  Data<float2> vt_c(4);
  Data<double2> vt_z(4);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, s_s.d_data, CUDA_R_32F, u_s.d_data, 2, CUDA_R_32F, vt_s.d_data, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, s_d.d_data, CUDA_R_64F, u_d.d_data, 2, CUDA_R_64F, vt_d.d_data, 2, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, s_c.d_data, CUDA_C_32F, u_c.d_data, 2, CUDA_C_32F, vt_c.d_data, 2, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, s_z.d_data, CUDA_C_64F, u_z.d_data, 2, CUDA_C_64F, vt_z.d_data, 2, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);
  cudaMalloc(&host_ws_s, host_ws_size_s);
  cudaMalloc(&host_ws_d, host_ws_size_d);
  cudaMalloc(&host_ws_c, host_ws_size_c);
  cudaMalloc(&host_ws_z, host_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, s_s.d_data, CUDA_R_32F, u_s.d_data, 2, CUDA_R_32F, vt_s.d_data, 2, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, s_d.d_data, CUDA_R_64F, u_d.d_data, 2, CUDA_R_64F, vt_d.d_data, 2, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, s_c.d_data, CUDA_C_32F, u_c.d_data, 2, CUDA_C_32F, vt_c.d_data, 2, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, s_z.d_data, CUDA_C_64F, u_z.d_data, 2, CUDA_C_64F, vt_z.d_data, 2, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  s_s.D2H();
  s_d.D2H();
  s_c.D2H();
  s_z.D2H();

  u_s.D2H();
  u_d.D2H();
  u_c.D2H();
  u_z.D2H();

  vt_s.D2H();
  vt_d.D2H();
  vt_c.D2H();
  vt_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(host_ws_s);
  cudaFree(host_ws_d);
  cudaFree(host_ws_c);
  cudaFree(host_ws_z);
  cudaFree(info);

  float expect_s[2] = {5.464985,0.365966};
  float expect_u[4] = {0.576048,0.817416,-0.817416,0.576048};
  float expect_vt[4] = {0.404554,0.914514,0.914514,-0.404554};

  if (compare_result(expect_s, s_s.h_data, 2) &&
      compare_result(expect_s, s_d.h_data, 2) &&
      compare_result(expect_s, s_c.h_data, 2) &&
      compare_result(expect_s, s_z.h_data, 2) &&
      compare_result(expect_u, u_s.h_data, 4) &&
      compare_result(expect_u, u_d.h_data, 4) &&
      compare_result(expect_u, u_c.h_data, 4) &&
      compare_result(expect_u, u_z.h_data, 4) &&
      compare_result(expect_vt, vt_s.h_data, 4) &&
      compare_result(expect_vt, vt_d.h_data, 4) &&
      compare_result(expect_vt, vt_c.h_data, 4) &&
      compare_result(expect_vt, vt_z.h_data, 4))
    printf("DnXgesvd pass\n");
  else {
    printf("DnXgesvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGesvd() {
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);

  Data<float> s_s(2);
  Data<double> s_d(2);
  Data<float> s_c(2);
  Data<double> s_z(2);

  Data<float> u_s(4);
  Data<double> u_d(4);
  Data<float2> u_c(4);
  Data<double2> u_z(4);

  Data<float> vt_s(4);
  Data<double> vt_d(4);
  Data<float2> vt_c(4);
  Data<double2> vt_z(4);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, s_s.d_data, CUDA_R_32F, u_s.d_data, 2, CUDA_R_32F, vt_s.d_data, 2, CUDA_R_32F, &device_ws_size_s);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, s_d.d_data, CUDA_R_64F, u_d.d_data, 2, CUDA_R_64F, vt_d.d_data, 2, CUDA_R_64F, &device_ws_size_d);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, s_c.d_data, CUDA_C_32F, u_c.d_data, 2, CUDA_C_32F, vt_c.d_data, 2, CUDA_C_32F, &device_ws_size_c);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, s_z.d_data, CUDA_C_64F, u_z.d_data, 2, CUDA_C_64F, vt_z.d_data, 2, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, s_s.d_data, CUDA_R_32F, u_s.d_data, 2, CUDA_R_32F, vt_s.d_data, 2, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, s_d.d_data, CUDA_R_64F, u_d.d_data, 2, CUDA_R_64F, vt_d.d_data, 2, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, s_c.d_data, CUDA_C_32F, u_c.d_data, 2, CUDA_C_32F, vt_c.d_data, 2, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, s_z.d_data, CUDA_C_64F, u_z.d_data, 2, CUDA_C_64F, vt_z.d_data, 2, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  s_s.D2H();
  s_d.D2H();
  s_c.D2H();
  s_z.D2H();

  u_s.D2H();
  u_d.D2H();
  u_c.D2H();
  u_z.D2H();

  vt_s.D2H();
  vt_d.D2H();
  vt_c.D2H();
  vt_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_s[2] = {5.464985,0.365966};
  float expect_u[4] = {0.576048,0.817416,-0.817416,0.576048};
  float expect_vt[4] = {0.404554,0.914514,0.914514,-0.404554};

  if (compare_result(expect_s, s_s.h_data, 2) &&
      compare_result(expect_s, s_d.h_data, 2) &&
      compare_result(expect_s, s_c.h_data, 2) &&
      compare_result(expect_s, s_z.h_data, 2) &&
      compare_result(expect_u, u_s.h_data, 4) &&
      compare_result(expect_u, u_d.h_data, 4) &&
      compare_result(expect_u, u_c.h_data, 4) &&
      compare_result(expect_u, u_z.h_data, 4) &&
      compare_result(expect_vt, vt_s.h_data, 4) &&
      compare_result(expect_vt, vt_d.h_data, 4) &&
      compare_result(expect_vt, vt_c.h_data, 4) &&
      compare_result(expect_vt, vt_z.h_data, 4))
    printf("DnGesvd pass\n");
  else {
    printf("DnGesvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXpotrf() {
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float2> a_c(a.data(), 9);
  Data<double2> a_z(a.data(), 9);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c.d_data, 3, CUDA_R_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z.d_data, 3, CUDA_R_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);
  cudaMalloc(&host_ws_s, host_ws_size_s);
  cudaMalloc(&host_ws_d, host_ws_size_d);
  cudaMalloc(&host_ws_c, host_ws_size_c);
  cudaMalloc(&host_ws_z, host_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c.d_data, 3, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z.d_data, 3, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(host_ws_s);
  cudaFree(host_ws_d);
  cudaFree(host_ws_c);
  cudaFree(host_ws_z);
  cudaFree(info);

  float expect_a[9] = {1.414214,-0.707107,0.000000,-1.000000,1.224745,-0.816497,0.000000,-1.000000,1.154701};
  if (compare_result(expect_a, a_s.h_data, 9) &&
      compare_result(expect_a, a_d.h_data, 9) &&
      compare_result(expect_a, a_c.h_data, 9) &&
      compare_result(expect_a, a_z.h_data, 9))
    printf("DnXpotrf pass\n");
  else {
    printf("DnXpotrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnPotrf() {
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float2> a_c(a.data(), 9);
  Data<double2> a_z(a.data(), 9);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, &device_ws_size_s);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, &device_ws_size_d);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c.d_data, 3, CUDA_R_32F, &device_ws_size_c);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z.d_data, 3, CUDA_R_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  cudaMalloc(&device_ws_s, device_ws_size_s);
  cudaMalloc(&device_ws_d, device_ws_size_d);
  cudaMalloc(&device_ws_c, device_ws_size_c);
  cudaMalloc(&device_ws_z, device_ws_size_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c.d_data, 3, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z.d_data, 3, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[9] = {1.414214,-0.707107,0.000000,-1.000000,1.224745,-0.816497,0.000000,-1.000000,1.154701};
  if (compare_result(expect_a, a_s.h_data, 9) &&
      compare_result(expect_a, a_d.h_data, 9) &&
      compare_result(expect_a, a_c.h_data, 9) &&
      compare_result(expect_a, a_z.h_data, 9))
    printf("DnPotrf pass\n");
  else {
    printf("DnPotrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXpotrs() {
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float2> a_c(a.data(), 9);
  Data<double2> a_z(a.data(), 9);
  std::vector<float> b = {0, 0, 4};
  Data<float> b_s(b.data(), 3);
  Data<double> b_d(b.data(), 3);
  Data<float2> b_c(b.data(), 3);
  Data<double2> b_z(b.data(), 3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, b_s.d_data, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, b_d.d_data, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_32F, a_c.d_data, 3, CUDA_C_32F, b_c.d_data, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_64F, a_z.d_data, 3, CUDA_C_64F, b_z.d_data, 3, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(info);

  float expect_b[3] = {1,2,3};
  if (compare_result(expect_b, b_s.h_data, 3) &&
      compare_result(expect_b, b_d.h_data, 3) &&
      compare_result(expect_b, b_c.h_data, 3) &&
      compare_result(expect_b, b_z.h_data, 3))
    printf("DnXpotrs pass\n");
  else {
    printf("DnXpotrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnPotrs() {
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float2> a_c(a.data(), 9);
  Data<double2> a_z(a.data(), 9);
  std::vector<float> b = {0, 0, 4};
  Data<float> b_s(b.data(), 3);
  Data<double> b_d(b.data(), 3);
  Data<float2> b_c(b.data(), 3);
  Data<double2> b_z(b.data(), 3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_32F, a_s.d_data, 3, CUDA_R_32F, b_s.d_data, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_64F, a_d.d_data, 3, CUDA_R_64F, b_d.d_data, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_32F, a_c.d_data, 3, CUDA_C_32F, b_c.d_data, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_64F, a_z.d_data, 3, CUDA_C_64F, b_z.d_data, 3, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(info);

  float expect_b[3] = {1,2,3};
  if (compare_result(expect_b, b_s.h_data, 3) &&
      compare_result(expect_b, b_d.h_data, 3) &&
      compare_result(expect_b, b_c.h_data, 3) &&
      compare_result(expect_b, b_z.h_data, 3))
    printf("DnPotrs pass\n");
  else {
    printf("DnPotrs fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusolverDnXgetrf();
  test_cusolverDnXgetrfnp();
  test_cusolverDnGetrf();
  test_cusolverDnXgetrs();
  test_cusolverDnGetrs();
  test_cusolverDnXgeqrf();
  test_cusolverDnGeqrf();
  test_cusolverDnXgesvd();
  test_cusolverDnGesvd();
  test_cusolverDnXpotrf();
  test_cusolverDnPotrf();
  test_cusolverDnXpotrs();
  test_cusolverDnPotrs();

  if (test_passed)
    return 0;
  return -1;
}
