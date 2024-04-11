// ===------------ bindless_images.cpp ---------- -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <dpct/dpct.hpp>

dpct::experimental::bindless_image_wrapper<sycl::float4, 4> i;

int main() {
  auto q = dpct::get_default_queue();
  dpct::image_channel c = dpct::image_channel::create<sycl::float4>();
  dpct::sampling_info samp;
  auto f = (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4), q);
  size_t p, x = 1, y = 1, z = 1;
  auto f2d = (sycl::float4 *)dpct::dpct_malloc(p, sizeof(sycl::float4), 1);
  printf("prepare variable pass!\n");
  dpct::experimental::image_mem_wrapper_ptr m;
  m = new dpct::experimental::image_mem_wrapper(c, sycl::range(1));
  m = new dpct::experimental::image_mem_wrapper(c, x);
  m = new dpct::experimental::image_mem_wrapper(c, x, y);
  m = new dpct::experimental::image_mem_wrapper(c, x, y, z);
  m->get_channel();
  m->get_range();
  m->get_desc();
  m->get_handle();
  dpct::image_data d; //(m); // TODO: need support.
  d.set_data(m);
  printf("image mem wrapper pass!\n");
  dpct::experimental::async_dpct_memcpy(m, 0, 0, f, 1, 1, 1, q);
  dpct::experimental::dpct_memcpy(m, 0, 0, f, 1, 1, 1, q);
  dpct::experimental::async_dpct_memcpy(m, 0, 0, f, 1, q);
  dpct::experimental::dpct_memcpy(m, 0, 0, f, 1, q);
  dpct::experimental::async_dpct_memcpy(f, m, 0, 0, 1, 1, 1, q);
  dpct::experimental::dpct_memcpy(f, m, 0, 0, 1, 1, 1, q);
  dpct::experimental::async_dpct_memcpy(f, m, 0, 0, 1, q);
  dpct::experimental::dpct_memcpy(f, m, 0, 0, 1, q);
  dpct::experimental::dpct_memcpy(m, 0, 0, m, 0, 0, 1, 1, q);
  dpct::experimental::dpct_memcpy(m, 0, 0, m, 0, 0, 1, q);
  printf("memory copy pass!\n");
  auto h = dpct::experimental::create_bindless_image(d, samp, q);
  dpct::experimental::get_data(h);
  dpct::experimental::get_sampling_info(h);
  dpct::experimental::destroy_bindless_image(h, q);
  printf("texture object pass!\n");
  i.attach(f, 1, c, q);
  i.attach(f, 1, q);
  i.attach(f2d, 1, 1, p, c, q);
  i.attach(f2d, 1, 1, p, c, q);
  i.attach(m, c, q);
  i.attach(m, q);
  i.detach(q);
  i.set_channel(c);
  i.get_channel();
  i.set_channel_size(4, 1);
  i.get_channel_size();
  i.set_channel_data_type(dpct::image_channel_data_type::fp);
  i.get_channel_data_type();
  i.set(sycl::addressing_mode::repeat);
  i.get_addressing_mode();
  i.set(sycl::coordinate_normalization_mode::unnormalized);
  i.is_coordinate_normalized();
  i.set(sycl::filtering_mode::nearest);
  i.get_filtering_mode();
  i.get_handle();
  printf("texture referece pass!\n");
  delete m;
  printf("test pass!\n");
  return 0;
}
