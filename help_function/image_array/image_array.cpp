// ====------ image_array.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define C2S_NAMED_LAMBDA
#include <dpct/dpct.hpp>

dpct::image_wrapper<sycl::float4, 3, true> tex43;
dpct::image_wrapper<sycl::float4, 2, true> tex42;

void test_image(sycl::float4* out, dpct::image_accessor_ext<sycl::float4, 2,true> acc42,
                  dpct::image_accessor_ext<sycl::float4, 1,true> acc21) {
  out[0] = acc42.read(16, 0.5f, 0.5f);
  sycl::float4 data32 = acc21.read(16, 0.5f);
  out[1].x() = data32.x();
  out[1].y() = data32.y();
}

int main() {

  sycl::float4 *host_buffer = new sycl::float4[640 * 480 * 24];

  for(int i = 0; i < 640 * 480 * 24; ++i) {
	  host_buffer[i] = sycl::float4{10.0f, 10.0f, 10.0f, 10.0f};
  }
  sycl::float4 *device_buffer;
  device_buffer = (sycl::float4 *)dpct::dpct_malloc(
                      640 * 480 * 24 * sizeof(sycl::float4));
  dpct::dpct_memcpy(device_buffer, host_buffer, 640 * 480 * 24 * sizeof(sycl::float4));

  dpct::image_channel chn2 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  dpct::image_channel chn4 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  chn4.set_channel_size(4, 32);

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array2;
  dpct::image_matrix_p array3;
 
  array2 = new dpct::image_matrix(chn2, sycl::range<2>(640, 480));
  array3 = new dpct::image_matrix(chn4, sycl::range<3>(640, 480, 24));

  dpct::dpct_memcpy(array2->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(sycl::float4), 640 * 480 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(sycl::float4), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(sycl::float4), 640 * 480 * 24 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(sycl::float4), 1, 1));

  dpct::image_wrapper_base *tex42;
  dpct::image_data res22;
  dpct::sampling_info texDesc22;
  res22.set_data(array2);

  tex43.attach(array3);

  tex43.set(sycl::addressing_mode::clamp);
  texDesc22.set(sycl::addressing_mode::clamp);
  tex43.set(sycl::coordinate_normalization_mode::normalized);

  texDesc22.set(sycl::coordinate_normalization_mode::normalized);

  tex43.set(sycl::filtering_mode::linear);
  texDesc22.set(sycl::filtering_mode::linear);

  tex42 = dpct::create_image_wrapper(res22, texDesc22);

  sycl::float4 d[32];
  for(int i = 0; i < 32; ++i) {
	  d[i] = sycl::float4{1.0f, 1.0f, 1.0f, 1.0f};
  }
  {
    sycl::buffer<sycl::float4, 1> buf(d, sycl::range<1>(32));
    tex43.create_image();
    static_cast<dpct::image_wrapper<sycl::float4, 2, true> *>(tex42)->create_image();
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto acc42 = tex43.get_access(cgh);
      auto acc21 = static_cast<dpct::image_wrapper<sycl::float4, 2, true> *>(tex42)->get_access(cgh);

      auto smpl42 = tex43.get_sampler();
      auto smpl21 = tex42->get_sampler();

      auto acc_out = buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);

      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>([=] {
        test_image(acc_out.get_pointer(),dpct::image_accessor_ext<sycl::float4, 2, true>(smpl42, acc42),
                   dpct::image_accessor_ext<sycl::float4, 1, true>(smpl21, acc21));
      });
    });
  }

  printf("d[0]: x[%f] y[%f] z[%f] w[%f]\n", d[0].x(), d[0].y(), d[0].z(), d[0].w());
  printf("d[1]: x[%f] y[%f] z[%f] w[%f]\n", d[1].x(), d[1].y(), d[1].z(), d[1].w());

  tex43.detach();

  delete tex42;
}
