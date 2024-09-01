// ====------ image_wrapper.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVLE_NONE
#define DPCT_NAMED_LAMBDA
#include <dpct/dpct.hpp>

dpct::image_wrapper<sycl::float4, 2> tex42;
dpct::image_wrapper<sycl::int4, 1> tex_int4;
dpct::image_wrapper<sycl::uint4, 3> tex43;

// test_feature:image_accessor_ext
void test_image(sycl::float4* out, dpct::image_accessor_ext<sycl::float4, 2> acc42,
                  dpct::image_accessor_ext<sycl::int4, 1> acc21,
                  dpct::image_accessor_ext<sycl::uint4, 3> acc13) {
  // test_feature:read
  out[0] = acc42.read(0.5f, 0.5f);
  // test_feature:read
  sycl::uint4 data13 = acc13.read((uint64_t)3, (int)4, (short)5);
  // test_feature:read
  sycl::int4 data21 = acc21.read(0.5f);
  int a = data21.x();
  int b = data13.x();
  out[1] = {a,b,a,b};
}

int main() {

  sycl::float4 *host_buffer = new sycl::float4[640 * 480 * 24];
  sycl::int4 *host_char_buffer = new sycl::int4[640];

  for(int i = 0; i < 640 * 480 * 24; ++i) {
	  host_buffer[i] = sycl::float4{10.0f, 10.0f, 10.0f, 10.0f};
	  if (i < 640)
		  host_char_buffer[i] = sycl::int4{30, 30, 30, 30};
  }
  sycl::float4 *device_buffer;
  device_buffer = (sycl::float4 *)dpct::dpct_malloc(
                      640 * 480 * 24 * sizeof(sycl::float4));
  dpct::dpct_memcpy(device_buffer, host_buffer, 640 * 480 * 24 * sizeof(sycl::float4));

  dpct::image_channel chn1 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::unsigned_int);
  dpct::image_channel chn2 = dpct::image_channel::create<sycl::int4>();
  dpct::image_channel chn4 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  chn4.set_channel_size(4, 32);

  sycl::float4 *image_data2 = (sycl::float4 *)std::malloc(650 * 480 * sizeof(sycl::float4));

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array3;

  array1 = new dpct::image_matrix(chn2, sycl::range<1>(640));
  array3 = new dpct::image_matrix(chn1, sycl::range<3>(640, 480, 24));

  tex42.attach(image_data2, 640, 480, 650 * sizeof(sycl::float4));

  dpct::dpct_memcpy(array1->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * sizeof(sycl::int4), 640 * sizeof(sycl::int4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * sizeof(sycl::int4), 1, 1));
  dpct::dpct_memcpy(dpct::pitched_data(image_data2, 650 * sizeof(sycl::float4), 640 * sizeof(sycl::float4*), 480), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(sycl::float4), 640 * 480 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(sycl::float4), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(sycl::uint4), 640 * 480 * 24 * sizeof(sycl::uint4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(sycl::uint4), 1, 1));

  tex43.attach(array3);

  dpct::image_wrapper_base *tex41;
  dpct::image_data res21;
  dpct::sampling_info texDesc21;
  res21.set_data(array1);

  tex42.set(sycl::addressing_mode::clamp);
  texDesc21.set(sycl::addressing_mode::clamp);
  tex43.set(sycl::addressing_mode::clamp);

  tex42.set(sycl::coordinate_normalization_mode::normalized);
  texDesc21.set_coordinate_normalization_mode(1);
  tex43.set(sycl::coordinate_normalization_mode::unnormalized);

  tex42.set(sycl::filtering_mode::linear);
  texDesc21.set(sycl::filtering_mode::linear);
  tex43.set(sycl::filtering_mode::linear);

  tex42.set(sycl::addressing_mode::clamp, sycl::filtering_mode::linear, 1);
  tex42.set(sycl::addressing_mode::clamp, sycl::filtering_mode::linear, sycl::coordinate_normalization_mode::normalized);

  texDesc21.set(sycl::addressing_mode::clamp, sycl::filtering_mode::linear, 1);
  texDesc21.set(sycl::addressing_mode::clamp, sycl::filtering_mode::linear, sycl::coordinate_normalization_mode::normalized);

  tex41 = dpct::create_image_wrapper(res21, texDesc21);

  sycl::float4 d[32];
  for(int i = 0; i < 32; ++i) {
	  d[i] = sycl::float4{1.0f, 1.0f, 1.0f, 1.0f};
  }
  {
    sycl::buffer<sycl::float4, 1> buf(d, sycl::range<1>(32));
    tex42.create_image();
    tex43.create_image();
    static_cast<dpct::image_wrapper<sycl::int4, 1> *>(tex41)->create_image();
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto acc42 = tex42.get_access(cgh);
      auto acc13 = tex43.get_access(cgh);
      auto acc21 = static_cast<dpct::image_wrapper<sycl::int4, 1> *>(tex41)->get_access(cgh);

      auto smpl42 = tex42.get_sampler();
      auto smpl13 = tex43.get_sampler();
      auto smpl21 = tex41->get_sampler();

      auto acc_out = buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);

      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>([=] {
        test_image(acc_out.get_pointer(),dpct::image_accessor_ext<sycl::float4, 2>(smpl42, acc42),
                   dpct::image_accessor_ext<sycl::int4, 1>(smpl21, acc21),
                   dpct::image_accessor_ext<sycl::uint4, 3>(smpl13, acc13));
      });
    });
  }

  printf("d[0]: x[%f] y[%f] z[%f] w[%f]\n", d[0].x(), d[0].y(), d[0].z(), d[0].w());
  printf("d[1]: x[%f] y[%f] z[%f] w[%f]\n", d[1].x(), d[1].y(), d[1].z(), d[1].w());

  tex42.detach();
  tex43.detach();

  delete tex41;
  sycl::free(device_buffer, dpct::get_default_queue());
  std::free(host_buffer);
  std::free(host_char_buffer);
  std::free(image_data2);
}
