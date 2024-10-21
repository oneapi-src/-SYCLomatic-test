// ====------ image_image_wrapper.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

int main() {
  dpct::image_channel IC;
  sycl::range<3> Range(1,1,1);
  // test_feature:image_matrix
  dpct::image_matrix IM(IC, Range);
  // test_feature:image_data
  dpct::image_data ID(&IM);
  dpct::sampling_info SI;

  // test_feature:image_wrapper_base()
  create_image_wrapper(ID, SI);

  // image_wrapper
  // test_feature:image_wrapper
  dpct::image_wrapper<sycl::float4, 3> tex43;
  dpct::image_channel chn1 =
    dpct::image_channel(16, 16, 16, 16, dpct::image_channel_data_type::unsigned_int);
  dpct::image_matrix_p array3;
  array3 = new dpct::image_matrix(chn1, sycl::range<3>(640, 480, 24));

  void *data;
  size_t count;
  // test_feature:attach(void *, size_t)
  tex43.attach(data, count);

  // test_feature:attach(void *, size_t, image_channel)
  tex43.attach(data, count, chn1);

  // test_feature:attach(void *, size_t, size_t, size_t)
  tex43.attach(data, count, 1, 1);

  // test_feature:attach(void *, size_t, size_t, size_t, image_channel)
  tex43.attach(data, count, 1, 1, chn1);

  // test_feature:attach(image_matrix *)
  tex43.attach(array3);

  // test_feature:detach()
  tex43.detach();

  // test_feature:attach(image_matrix *, image_channel)
  tex43.attach(array3, chn1);
  tex43.create_image();
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    // test_feature:get_access
    tex43.get_access(cgh);
  });

  // image_wrapper_base
  dpct::image_wrapper_base *base=&tex43;
  // test_feature:attach(image_data)
  base->attach(ID);

  // test_feature:get_sampling_info()
  base->get_sampling_info();

  // test_feature:get_data()
  base->get_data();

  // test_feature:set_data(image_data)
  base->set_data(ID);

  // test_feature:get_channel()
  base->get_channel();

  // test_feature:set_channel(image_channel)
  base->set_channel(chn1);

  // test_feature:get_channel_data_type()
  auto CDT = base->get_channel_data_type();

  // test_feature:set_channel_data_type(image_channel_data_type)
  base->set_channel_data_type(CDT);

  // test_feature:get_channel_size()
  auto CS = base->get_channel_size();

  // test_feature:set_channel_size(unsigned, unsigned)
  base->set_channel_size(1,1);

  // test_feature:get_addressing_mode()
  auto AM = base->get_addressing_mode();

  // test_feature:set(addressing_mode)
  base->set(AM);

  // test_feature:get_filtering_mode()
  auto FM = base->get_filtering_mode();

  // test_feature:set(filtering_mode)
  base->set(FM);

  // test_feature:get_coordinate_normalization_mode()
  auto CNM = base->get_coordinate_normalization_mode();

  // test_feature:set(CNM)
  base->set(CNM);

  // test_feature:is_coordinate_normalized()
  auto ICN = base->is_coordinate_normalized();

  // test_feature:set_coordinate_normalization_mode(bool)
  base->set_coordinate_normalization_mode(ICN);

  // test_feature:set(addressing_mode, filtering_mode, coordinate_normalization_mode)
  base->set(AM, FM, CNM);

  // test_feature:set(addressing_mode, filtering_mode, is_normalized)
  base->set(AM, FM, 0);

  // test_feature:get_channel_num()
  auto CN = base->get_channel_num();

  // test_feature:set_channel_num()
  base->set_channel_num(CN);

  // test_feature:get_channel_type()
  auto CT = base->get_channel_type();

  // test_feature:set_channel_type()
  base->set_channel_type(CT);

  // test_feature:get_sampler()
  base->get_sampler();

  return 0;
}
