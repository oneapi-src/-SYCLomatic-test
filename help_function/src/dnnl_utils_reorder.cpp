// ====------ dnnl_utils_reorder.cpp --------------===//
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
// test_feature:reorder
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
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;

    handle.create_engine();

    sycl::queue *stream1;
    stream1 = dev_ct1.create_queue();
    handle.set_queue(stream1);

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    outTensor.set(dpct::dnnl::memory_format_tag::nhwc, T, n, c, h, w);

    HT *data, *out;
    std::vector<HT> host_data(ele_num, 0);
    std::vector<HT> host_out(ele_num, 0);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
        host_out[i] = 0;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num * sizeof(HT)).wait();

    float alpha = 3.f, beta = 1.f;

    auto s = (handle.async_reorder(alpha, dataTensor, data, beta, outTensor, out), 0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    std::vector<float> expect = {
      0, 75, 3, 78, 6,
      81, 9, 84, 12, 87,
      15, 90, 18, 93, 21,
      96, 24, 99, 27, 102,
      30, 105, 33, 108, 36,
      111, 39, 114, 42, 117,
      45, 120, 48, 123, 51,
      126, 54, 129, 57, 132,
      60, 135, 63, 138, 66,
      141, 69, 144, 72, 147
    };
    check(expect, host_out, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
}

int main() {
    test<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}