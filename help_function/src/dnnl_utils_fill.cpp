// ====------ dnnl_utils_fill.cpp -----------------===//
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
// test_feature:fill
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
    HT * data;
    HT value = 1.5;
    std::vector<HT> host_data(ele_num, 0);

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    handle.async_fill(dataTensor, data, &value);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_data.data(), data, ele_num * sizeof(HT)).wait();
    float precision = 1e-3;
    for(int i = 0; i < ele_num; i++) {
        if(std::abs(host_data[i] -value) > precision) {
            std::cout << "test fail" << std::endl;
            exit(-1);
        } 
    }
}

int main() {
    test<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}