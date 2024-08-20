// ====------ dnnl_utils_scale.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dnnl_utils.hpp>

#include <iostream>
#include <vector>
// test_feature:engine_ext
// test_feature:memory_format_tag
// test_feature:memory_desc_ext
// test_feature:scale
template <dpct::library_data_t T> struct dt_trait {
    typedef void type;
};
template <> struct dt_trait<dpct::library_data_t::real_float> {
    typedef float type;
};

template <> struct dt_trait<dpct::library_data_t::real_int32> {
    typedef int type;
};
template <> struct dt_trait<dpct::library_data_t::real_half> {
    typedef float type;
};

template<typename T>
void check(std::vector<T> &expect, std::vector<T> &actual, int num, float precision) {
  for(int i = 0; i < num; i++){
      if(std::abs(expect[i] - actual[i]) > precision) {
          std::cout << "test failed" << std::endl;
          std::cout << "expect:" << expect[i] << std::endl;
          std::cout << "actual:" << actual[i] << std::endl;
          exit(-1);
      }
  }
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor;

    handle.create_engine();

    sycl::queue *stream1;
    stream1 = dev_ct1.create_queue();
    handle.set_queue(stream1);

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

    HT *data;
    std::vector<HT> host_data(ele_num);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();

    float alpha = 3.f;

    auto s = (handle.scale(alpha, dataTensor, data), 0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_data.data(), data, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        0, 3, 6, 9, 12,
        15, 18, 21, 24, 27,
        30, 33, 36, 39, 42,
        45, 48, 51, 54, 57,
        60, 63, 66, 69, 72,
        75, 78, 81, 84, 87,
        90, 93, 96, 99, 102,
        105, 108, 111, 114, 117,
        120, 123, 126, 129, 132,
        135, 138, 141, 144, 147
      };
    check(expect, host_data, expect.size(), 1e-3);

    sycl::free(data, *stream1);
}

int main() {
    test<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}