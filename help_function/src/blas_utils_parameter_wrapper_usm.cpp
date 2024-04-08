// ===------- blas_utils_parameter_wrapper_usm.cpp ----- -*- C++ -* ------=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <cstdio>

bool pass = true;

void test_iamax1() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::queue_ptr handle;
  handle = &q_ct1;
  float *x;
  x = sycl::malloc_shared<float>(5, q_ct1);
  x[0] = 30;
  x[1] = 40;
  x[2] = 53;
  x[3] = 1;
  x[4] = 100;
  int64_t result1;
  [&]() {
    dpct::blas::wrapper_int64_out res(*handle, &result1);
    oneapi::mkl::blas::column_major::iamax(*handle, 5, x, 1, res.get_ptr(),
                                           oneapi::mkl::index_base::one);
  }();
  int64_t result2;
  int64_t *result2_d;
  result2_d = sycl::malloc_device<int64_t>(1, q_ct1);
  [&]() {
    dpct::blas::wrapper_int64_out res(*handle, result2_d);
    oneapi::mkl::blas::column_major::iamax(*handle, 5, x, 1, res.get_ptr(),
                                           oneapi::mkl::index_base::one);
  }();
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(&result2, result2_d, sizeof(int64_t)).wait();
  handle = nullptr;
  sycl::free(x, q_ct1);
  sycl::free(result2_d, q_ct1);
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  dpct::queue_ptr handle;
  handle = &q_ct1;
  float *x;
  x = sycl::malloc_shared<float>(5, q_ct1);
  x[0] = 30;
  x[1] = 40;
  x[2] = 53;
  x[3] = 1;
  x[4] = 100;
  int result1;
  [&]() {
    dpct::blas::wrapper_int_to_int64_out res(*handle, &result1);
    oneapi::mkl::blas::column_major::iamax(*handle, 5, x, 1, res.get_ptr(),
                                           oneapi::mkl::index_base::one);
  }();
  int result2;
  int *result2_d;
  result2_d = sycl::malloc_device<int>(1, q_ct1);
  [&]() {
    dpct::blas::wrapper_int_to_int64_out res(*handle, result2_d);
    oneapi::mkl::blas::column_major::iamax(*handle, 5, x, 1, res.get_ptr(),
                                           oneapi::mkl::index_base::one);
  }();
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(&result2, result2_d, sizeof(int)).wait();
  handle = nullptr;
  sycl::free(x, q_ct1);
  sycl::free(result2_d, q_ct1);
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
  handle = &dpct::get_default_queue();
  float *x = sycl::malloc_device<float>(4, *handle);
  float x_h[4];
  x_h[0] = 1;
  x_h[1] = 1.73205;
  handle->memcpy(x, x_h, 4 * sizeof(float)).wait();
  [&]() {
    dpct::blas::wrapper_float_inout a(*handle, x);
    dpct::blas::wrapper_float_inout b(*handle, x + 1);
    dpct::blas::wrapper_float_out c(*handle, x + 2);
    dpct::blas::wrapper_float_out s(*handle, x + 3);
    oneapi::mkl::blas::column_major::rotg(*handle, a.get_ptr(), b.get_ptr(),
                                          c.get_ptr(), s.get_ptr());
  }();
  dpct::get_current_device().queues_wait_and_throw();
  handle->memcpy(x_h, x, 4 * sizeof(float)).wait();
  handle = nullptr;
  if (std::abs(x_h[0] - 2.0f) < 0.01 && std::abs(x_h[1] - 2.0f) < 0.01 &&
      std::abs(x_h[2] - 0.5f) < 0.01 && std::abs(x_h[3] - 0.866025f) < 0.01) {
    printf("test_rotg1 pass\n");
  } else {
    printf("test_rotg1 fail:\n");
    printf("%f,%f,%f,%f\n", x_h[0], x_h[1], x_h[2], x_h[3]);
    pass = false;
  }
  sycl::free(x, dpct::get_default_queue());
}

void test_rotg2() {
  dpct::queue_ptr handle;
  handle = &dpct::get_default_queue();
  float *x = (float *)std::malloc(sizeof(float) * 4);
  x[0] = 1;
  x[1] = 1.73205;
  [&]() {
    dpct::blas::wrapper_float_inout a(*handle, x);
    dpct::blas::wrapper_float_inout b(*handle, x + 1);
    dpct::blas::wrapper_float_out c(*handle, x + 2);
    dpct::blas::wrapper_float_out s(*handle, x + 3);
    oneapi::mkl::blas::column_major::rotg(*handle, a.get_ptr(), b.get_ptr(),
                                          c.get_ptr(), s.get_ptr());
  }();
  dpct::get_current_device().queues_wait_and_throw();
  handle = nullptr;
  if (std::abs(x[0] - 2.0f) < 0.01 && std::abs(x[1] - 2.0f) < 0.01 &&
      std::abs(x[2] - 0.5f) < 0.01 && std::abs(x[3] - 0.866025f) < 0.01) {
    printf("test_rotg2 pass\n");
  } else {
    printf("test_rotg2 fail:\n");
    printf("%f,%f,%f,%f\n", x[0], x[1], x[2], x[3]);
    pass = false;
  }
  std::free(x);
}

void test_rotm1() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float *x;
  float *y;
  x = sycl::malloc_shared<float>(1, q_ct1);
  y = sycl::malloc_shared<float>(1, q_ct1);
  *x = 1;
  *y = 2;
  float param[5];
  param[0] = -1;
  param[1] = 1;
  param[2] = 2;
  param[3] = 3;
  param[4] = 4;
  [&]() {
    dpct::blas::wrapper_float_in param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotm(*handle, 1, x, 1, y, 1,
                                          param_mem.get_ptr());
  }();
  q_ct1.wait();
  if (std::abs(*x - 7.0f) < 0.01 && std::abs(*y - 10.0f) < 0.01) {
    printf("test_rotm1 pass\n");
  } else {
    printf("test_rotm1 fail:\n");
    printf("%f,%f\n", *x, *y);
    pass = false;
  }
}

void test_rotm2() {
  dpct::queue_ptr handle;
  sycl::queue &q_ct1 = dpct::get_default_queue();
  handle = &q_ct1;
  float *x;
  float *y;
  x = sycl::malloc_shared<float>(1, q_ct1);
  y = sycl::malloc_shared<float>(1, q_ct1);
  *x = 1;
  *y = 2;
  float *param = sycl::malloc_device<float>(5, q_ct1);
  float param_h[5];
  param_h[0] = -1;
  param_h[1] = 1;
  param_h[2] = 2;
  param_h[3] = 3;
  param_h[4] = 4;
  q_ct1.memcpy(param, param_h, sizeof(float) * 5).wait();
  [&]() {
    dpct::blas::wrapper_float_in param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotm(*handle, 1, x, 1, y, 1,
                                          param_mem.get_ptr());
  }();
  q_ct1.wait();
  if (std::abs(*x - 7.0f) < 0.01 && std::abs(*y - 10.0f) < 0.01) {
    printf("test_rotm2 pass\n");
  } else {
    printf("test_rotm2 fail:\n");
    printf("%f,%f\n", *x, *y);
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
        *handle, d1_mem.get_ptr(), d2_mem.get_ptr(), x1_mem.get_ptr(),
        dpct::get_value<float>(&y1, *handle), param_mem.get_ptr());
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
  float four_values_h[4];
  float param_h[5];
  four_values_h[0] = 4;
  four_values_h[1] = 9;
  four_values_h[2] = 10;
  four_values_h[3] = 20;
  float *four_values = sycl::malloc_device<float>(4, q_ct1);
  q_ct1.memcpy(four_values, four_values_h, sizeof(float) * 4).wait();
  float *param = sycl::malloc_device<float>(5, q_ct1);
  [&]() {
    dpct::blas::wrapper_float_inout d1_mem(*handle, four_values);
    dpct::blas::wrapper_float_inout d2_mem(*handle, four_values + 1);
    dpct::blas::wrapper_float_inout x1_mem(*handle, four_values + 2);
    dpct::blas::wrapper_float_out param_mem(*handle, param, 5);
    oneapi::mkl::blas::column_major::rotmg(
        *handle, d1_mem.get_ptr(), d2_mem.get_ptr(), x1_mem.get_ptr(),
        dpct::get_value<float>(four_values + 3, *handle), param_mem.get_ptr());
  }();
  q_ct1.wait();
  q_ct1.memcpy(four_values_h, four_values, sizeof(float) * 4).wait();
  q_ct1.memcpy(param_h, param, sizeof(float) * 5).wait();
  if (std::abs(four_values_h[0] - 8.099999f) < 0.01 &&
      std::abs(four_values_h[1] - 3.6f) < 0.01 &&
      std::abs(four_values_h[2] - 22.222223f) < 0.01 &&
      std::abs(param_h[0] - 1.0f) < 0.01 &&
      std::abs(param_h[1] - 0.222222f) < 0.01 &&
      std::abs(param_h[4] - 0.5f) < 0.01) {
    printf("test_rotmg2 pass\n");
  } else {
    printf("test_rotmg2 fail:\n");
    printf("d1,d2,x1:%f,%f,%f\n", four_values_h[0], four_values_h[1],
           four_values_h[2]);
    printf("param:%f,%f,%f,%f,%f\n", param_h[0], param_h[1], param_h[2],
           param_h[3], param_h[4]);
    pass = false;
  }
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
