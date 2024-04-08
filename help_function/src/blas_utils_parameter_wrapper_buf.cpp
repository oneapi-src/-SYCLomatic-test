// ===------- blas_utils_parameter_wrapper_buf.cpp ----- -*- C++ -* ------=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <cstdio>

bool pass = true;

void test_iamax1() {
  dpct::queue_ptr handle;
  handle = &dpct::get_out_of_order_queue();
  float *x;
  x = (float *)dpct::dpct_malloc(sizeof(float) * 5);
  dpct::get_host_ptr<float>(x)[0] = 30;
  dpct::get_host_ptr<float>(x)[1] = 40;
  dpct::get_host_ptr<float>(x)[2] = 53;
  dpct::get_host_ptr<float>(x)[3] = 1;
  dpct::get_host_ptr<float>(x)[4] = 100;
  int64_t result1;
  [&]() {
    dpct::blas::wrapper_int64_out res(*handle, &result1);
    oneapi::mkl::blas::column_major::iamax(
        *handle, 5, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<int64_t>(res.get_ptr())),
        oneapi::mkl::index_base::one);
  }();
  int64_t result2;
  int64_t *result2_d;
  result2_d = (int64_t *)dpct::dpct_malloc(sizeof(int64_t));
  [&]() {
    dpct::blas::wrapper_int64_out res(*handle, result2_d);
    oneapi::mkl::blas::column_major::iamax(
        *handle, 5, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<int64_t>(res.get_ptr())),
        oneapi::mkl::index_base::one);
  }();
  dpct::get_current_device().queues_wait_and_throw();
  dpct::dpct_memcpy(&result2, result2_d, sizeof(int64_t), dpct::device_to_host);
  handle = nullptr;
  dpct::dpct_free(x);
  dpct::dpct_free(result2_d);
  if (result1 == 5 && result2 == 5) {
    printf("test_iamax1 pass\n");
  } else {
    printf("test_iamax1 fail:\n");
    printf("%ld\n", result1);
    printf("%ld\n", result2);
    pass = false;
  }
}

void test_iamax2() {
  dpct::queue_ptr handle;
  handle = &dpct::get_out_of_order_queue();
  float *x;
  x = (float *)dpct::dpct_malloc(sizeof(float) * 5);
  dpct::get_host_ptr<float>(x)[0] = 30;
  dpct::get_host_ptr<float>(x)[1] = 40;
  dpct::get_host_ptr<float>(x)[2] = 53;
  dpct::get_host_ptr<float>(x)[3] = 1;
  dpct::get_host_ptr<float>(x)[4] = 100;
  int result1;
  [&]() {
    dpct::blas::wrapper_int_to_int64_out res(*handle, &result1);
    oneapi::mkl::blas::column_major::iamax(
        *handle, 5, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<int64_t>(res.get_ptr())),
        oneapi::mkl::index_base::one);
  }();
  int result2;
  int *result2_d;
  result2_d = (int *)dpct::dpct_malloc(sizeof(int));
  [&]() {
    dpct::blas::wrapper_int_to_int64_out res(*handle, result2_d);
    oneapi::mkl::blas::column_major::iamax(
        *handle, 5, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<int64_t>(res.get_ptr())),
        oneapi::mkl::index_base::one);
  }();
  dpct::get_current_device().queues_wait_and_throw();
  dpct::dpct_memcpy(&result2, result2_d, sizeof(int), dpct::device_to_host);
  handle = nullptr;
  dpct::dpct_free(x);
  dpct::dpct_free(result2_d);
  if (result1 == 5 && result2 == 5) {
    printf("test_iamax2 pass\n");
  } else {
    printf("test_iamax2 fail:\n");
    printf("%d\n", result1);
    printf("%d\n", result2);
    pass = false;
  }
}

void test_rotg1() {
  dpct::queue_ptr handle;
  handle = &dpct::get_out_of_order_queue();
  float *a = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  float *b = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  float *c = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  float *s = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  dpct::get_host_ptr<float>(a)[0] = 1;
  dpct::get_host_ptr<float>(b)[0] = 1.73205;
  [&]() {
    dpct::blas::wrapper_float_inout a_m(*handle, a);
    dpct::blas::wrapper_float_inout b_m(*handle, b);
    dpct::blas::wrapper_float_out c_m(*handle, c);
    dpct::blas::wrapper_float_out s_m(*handle, s);
    oneapi::mkl::blas::column_major::rotg(
        *handle,
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(a_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(b_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(c_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(s_m.get_ptr())));
  }();
  dpct::get_current_device().queues_wait_and_throw();
  handle = nullptr;
  if (std::abs(dpct::get_host_ptr<float>(a)[0] - 2.0f) < 0.01 &&
      std::abs(dpct::get_host_ptr<float>(b)[0] - 2.0f) < 0.01 &&
      std::abs(dpct::get_host_ptr<float>(c)[0] - 0.5f) < 0.01 &&
      std::abs(dpct::get_host_ptr<float>(s)[0] - 0.866025f) < 0.01) {
    printf("test_rotg1 pass\n");
  } else {
    printf("test_rotg1 fail:\n");
    printf("%f,%f,%f,%f\n", dpct::get_host_ptr<float>(a)[0],
           dpct::get_host_ptr<float>(b)[0], dpct::get_host_ptr<float>(c)[0],
           dpct::get_host_ptr<float>(s)[0]);
    pass = false;
  }
  dpct::dpct_free(a);
  dpct::dpct_free(b);
  dpct::dpct_free(c);
  dpct::dpct_free(s);
}

void test_rotg2() {
  dpct::queue_ptr handle;
  handle = &dpct::get_out_of_order_queue();
  float *a = (float *)std::malloc(sizeof(float) * 1);
  float *b = (float *)std::malloc(sizeof(float) * 1);
  float *c = (float *)std::malloc(sizeof(float) * 1);
  float *s = (float *)std::malloc(sizeof(float) * 1);
  a[0] = 1;
  b[0] = 1.73205;
  [&]() {
    dpct::blas::wrapper_float_inout a_m(*handle, a);
    dpct::blas::wrapper_float_inout b_m(*handle, b);
    dpct::blas::wrapper_float_out c_m(*handle, c);
    dpct::blas::wrapper_float_out s_m(*handle, s);
    oneapi::mkl::blas::column_major::rotg(
        *handle,
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(a_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(b_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(c_m.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(s_m.get_ptr())));
  }();
  dpct::get_current_device().queues_wait_and_throw();
  handle = nullptr;
  if (std::abs(a[0] - 2.0f) < 0.01 && std::abs(b[0] - 2.0f) < 0.01 &&
      std::abs(c[0] - 0.5f) < 0.01 && std::abs(s[0] - 0.866025f) < 0.01) {
    printf("test_rotg2 pass\n");
  } else {
    printf("test_rotg2 fail:\n");
    printf("%f,%f,%f,%f\n", a[0], b[0], c[0], s[0]);
    pass = false;
  }
  std::free(a);
  std::free(b);
  std::free(c);
  std::free(s);
}

void test_rotm1() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float *x = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  float *y = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  dpct::get_host_ptr<float>(x)[0] = 1;
  dpct::get_host_ptr<float>(y)[0] = 2;
  float param[5];
  param[0] = -1;
  param[1] = 1;
  param[2] = 2;
  param[3] = 3;
  param[4] = 4;
  [&]() {
    dpct::blas::wrapper_float_in param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotm(
        *handle, 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y)), 1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(param_mem.get_ptr())));
  }();
  q_ct1.wait();
  if (std::abs(dpct::get_host_ptr<float>(x)[0] - 7.0f) < 0.01 &&
      std::abs(dpct::get_host_ptr<float>(y)[0] - 10.0f) < 0.01) {
    printf("test_rotm1 pass\n");
  } else {
    printf("test_rotm1 fail:\n");
    printf("%f,%f\n", dpct::get_host_ptr<float>(x)[0],
           dpct::get_host_ptr<float>(y)[0]);
    pass = false;
  }
}

void test_rotm2() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float *x = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  float *y = (float *)dpct::dpct_malloc(sizeof(float) * 1);
  dpct::get_host_ptr<float>(x)[0] = 1;
  dpct::get_host_ptr<float>(y)[0] = 2;
  float *param = (float *)dpct::dpct_malloc(sizeof(float) * 5);
  float param_h[5];
  param_h[0] = -1;
  param_h[1] = 1;
  param_h[2] = 2;
  param_h[3] = 3;
  param_h[4] = 4;
  dpct::dpct_memcpy(param, param_h, sizeof(float) * 5);
  [&]() {
    dpct::blas::wrapper_float_in param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotm(
        *handle, 1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(x)),
        1, dpct::rvalue_ref_to_lvalue_ref(dpct::get_buffer<float>(y)), 1,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(param_mem.get_ptr())));
  }();
  q_ct1.wait();
  if (std::abs(dpct::get_host_ptr<float>(x)[0] - 7.0f) < 0.01 &&
      std::abs(dpct::get_host_ptr<float>(y)[0] - 10.0f) < 0.01) {
    printf("test_rotm2 pass\n");
  } else {
    printf("test_rotm2 fail:\n");
    printf("%f,%f\n", dpct::get_host_ptr<float>(x)[0],
           dpct::get_host_ptr<float>(y)[0]);
    pass = false;
  }
}

void test_rotmg1() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float d1, d2, x1, y1;
  float param[5];
  d1 = 4;
  d2 = 9;
  x1 = 10;
  y1 = 20;
  [&]() {
    dpct::blas::wrapper_float_inout d1_mem(*handle, &d1);
    dpct::blas::wrapper_float_inout d2_mem(*handle, &d2);
    dpct::blas::wrapper_float_inout x1_mem(*handle, &x1);
    dpct::blas::wrapper_float_out param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotmg(
        *handle,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(d1_mem.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(d2_mem.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(x1_mem.get_ptr())),
        dpct::get_value<float>(&y1, *handle),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(param_mem.get_ptr())));
  }();
  q_ct1.wait();
  if (std::abs(d1 - 8.099999f) < 0.01 && std::abs(d2 - 3.6f) < 0.01 &&
      std::abs(x1 - 22.222223f) < 0.01 && std::abs(param[0] - 1.0f) < 0.01 &&
      std::abs(param[1] - 0.222222f) < 0.01 &&
      std::abs(param[4] - 0.5f) < 0.01) {
    printf("test_rotmg1 pass\n");
  } else {
    printf("test_rotmg1 fail:\n");
    printf("d1,d2,x1:%f,%f,%f\n", d1, d2, x1);
    printf("param:%f,%f,%f,%f,%f\n", param[0], param[1], param[2], param[3],
           param[4]);
    pass = false;
  }
}

void test_rotmg2() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float param_h[5];
  float d1 = 4;
  float d2 = 9;
  float x1 = 10;
  float y1 = 20;
  float *d1_d = (float *)dpct::dpct_malloc(sizeof(float));
  float *d2_d = (float *)dpct::dpct_malloc(sizeof(float));
  float *x1_d = (float *)dpct::dpct_malloc(sizeof(float));
  float *y1_d = (float *)dpct::dpct_malloc(sizeof(float));
  dpct::dpct_memcpy(d1_d, &d1, sizeof(float));
  dpct::dpct_memcpy(d2_d, &d2, sizeof(float));
  dpct::dpct_memcpy(x1_d, &x1, sizeof(float));
  dpct::dpct_memcpy(y1_d, &y1, sizeof(float));
  float *param = (float *)dpct::dpct_malloc(5 * sizeof(float));
  [&]() {
    dpct::blas::wrapper_float_inout d1_mem(*handle, d1_d);
    dpct::blas::wrapper_float_inout d2_mem(*handle, d2_d);
    dpct::blas::wrapper_float_inout x1_mem(*handle, x1_d);
    dpct::blas::wrapper_float_out param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotmg(
        *handle,
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(d1_mem.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(d2_mem.get_ptr())),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(x1_mem.get_ptr())),
        dpct::get_value<float>(y1_d, *handle),
        dpct::rvalue_ref_to_lvalue_ref(
            dpct::get_buffer<float>(param_mem.get_ptr())));
  }();
  q_ct1.wait();
  dpct::dpct_memcpy(&d1, d1_d, sizeof(float));
  dpct::dpct_memcpy(&d2, d2_d, sizeof(float));
  dpct::dpct_memcpy(&x1, x1_d, sizeof(float));
  dpct::dpct_memcpy(&y1, y1_d, sizeof(float));
  dpct::dpct_memcpy(param_h, param, sizeof(float) * 5);
  if (std::abs(d1 - 8.099999f) < 0.01 && std::abs(d2 - 3.6f) < 0.01 &&
      std::abs(x1 - 22.222223f) < 0.01 && std::abs(param_h[0] - 1.0f) < 0.01 &&
      std::abs(param_h[1] - 0.222222f) < 0.01 &&
      std::abs(param_h[4] - 0.5f) < 0.01) {
    printf("test_rotmg2 pass\n");
  } else {
    printf("test_rotmg2 fail:\n");
    printf("d1,d2,x1:%f,%f,%f\n", d1, d2, x1);
    printf("param:%f,%f,%f,%f,%f\n", param_h[0], param_h[1], param_h[2],
           param_h[3], param_h[4]);
    pass = false;
  }
  dpct::dpct_free(d1_d);
  dpct::dpct_free(d2_d);
  dpct::dpct_free(x1_d);
  dpct::dpct_free(y1_d);
  dpct::dpct_free(param);
}

int main() {
  test_iamax1();
  test_iamax2();
  test_rotg1();
  test_rotg2();
  test_rotm1();
  test_rotm2();
  test_rotmg1();
  test_rotmg2();
  if (pass)
    return 0;
  return -1;
}
