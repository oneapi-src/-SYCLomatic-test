// ====------ image_wrapper_2_usm.cpp -------------------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

// image data
// 0 1 2 3 4 5
// 0 1 2 3 4 5
// 0 1 2 3 4 5
// 0 1 2 3 4 5

void kernel(dpct::image_accessor_ext<sycl::float4, 2> texObj, float *res,
            sycl::nd_item<3> item_ct1) {
  int row = item_ct1.get_local_id(2);
  int col = item_ct1.get_local_id(1);
  sycl::float4 f4 = texObj.read(col, row);
  float val = f4.x();
  res[row * 6 + col] = val;
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  sycl::float4 data_h[24];
  memset(data_h, 0, sizeof(sycl::float4) * 24);
  data_h[0].x() = 0;
  data_h[1].x() = 1;
  data_h[2].x() = 2;
  data_h[3].x() = 3;
  data_h[4].x() = 4;
  data_h[5].x() = 5;
  memcpy(data_h + 6, data_h, sizeof(sycl::float4) * 6);
  memcpy(data_h + 12, data_h, sizeof(sycl::float4) * 6);
  memcpy(data_h + 18, data_h, sizeof(sycl::float4) * 6);

  sycl::float4 *data_d;
  data_d = sycl::malloc_device<sycl::float4>(24, q_ct1);
  q_ct1.memcpy(data_d, data_h, sizeof(sycl::float4) * 24).wait();

  dpct::image_data res;
  memset(&res, 0, sizeof(res));
  res.set_data_type(dpct::image_data_type::pitch);
  res.set_data_ptr(data_d);
  res.set_x(6);
  res.set_y(4);
  res.set_pitch(6 * sizeof(sycl::float4));
  res.set_channel(dpct::image_channel::create<sycl::float4>());
  dpct::sampling_info tex;
  memset(&tex, 0, sizeof(tex));
  tex.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest,
          sycl::coordinate_normalization_mode::unnormalized);
  dpct::image_wrapper_base_p texObj = 0;
  texObj = dpct::create_image_wrapper(res, tex);

  float *result;
  result = sycl::malloc_device<float>(24, q_ct1);
  static_cast<dpct::image_wrapper<sycl::float4, 2> *>(texObj)->create_image();
  q_ct1.submit([&](sycl::handler &cgh) {
    auto texObj_acc =
        static_cast<dpct::image_wrapper<sycl::float4, 2> *>(texObj)->get_access(
            cgh);

    auto texObj_smpl = texObj->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 6, 4), sycl::range<3>(1, 6, 4)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel(dpct::image_accessor_ext<sycl::float4, 2>(texObj_smpl,
                                                           texObj_acc),
                 result, item_ct1);
        });
  });
  float result_h[24];
  q_ct1.memcpy(&result_h, result, sizeof(float) * 24).wait();

  int sum = 0;
  for(int i = 0; i<24;i++)
    sum = sum + result_h[i];

  if (sum != 60) {
    printf("%f,%f,%f,%f,%f,%f\n", result_h[0], result_h[1], result_h[2], result_h[3], result_h[4], result_h[5]);
    printf("%f,%f,%f,%f,%f,%f\n", result_h[6], result_h[7], result_h[8], result_h[9], result_h[10], result_h[11]);
    printf("%f,%f,%f,%f,%f,%f\n", result_h[12], result_h[13], result_h[14], result_h[15], result_h[16], result_h[17]);
    printf("%f,%f,%f,%f,%f,%f\n", result_h[18], result_h[19], result_h[20], result_h[21], result_h[22], result_h[23]);
    return -1;
  }
  return 0;
}
