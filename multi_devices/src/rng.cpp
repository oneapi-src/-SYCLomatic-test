// ===------- rng.cpp ------------------------------------- *- C++ -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <sycl/sycl.hpp>
#include <dpct/rng_utils.hpp>
#include <cstdio>

int main() {
  sycl::queue cpu_q(sycl::cpu_selector_v, sycl::property::queue::in_order());
  sycl::queue gpu_q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  float *h_data = (float *)std::malloc(sizeof(float) * 10);
  float *d_data;
  d_data = sycl::malloc_shared<float>(10, gpu_q);

  dpct::rng::host_rng_ptr h_rng;
  dpct::rng::host_rng_ptr d_rng;

  h_rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mt2203, cpu_q);
  d_rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mt2203, gpu_q);
  h_rng->set_engine_idx(1);
  d_rng->set_engine_idx(1);

  h_rng->generate_gaussian(h_data, 10, 0, 1);
  d_rng->generate_gaussian(d_data, 10, 0, 1);

  cpu_q.wait();
  gpu_q.wait();

  for (int i = 0; i < 10; i++)
    printf("%f, ", h_data[i]);
  printf("\n");
  for (int i = 0; i < 10; i++)
    printf("%f, ", d_data[i]);
  printf("\n");

  for (int i = 0; i < 10; i++) {
    if (std::abs(h_data[i] - d_data[i]) > 0.01f)
      return -1;
  }

  return 0;
}
