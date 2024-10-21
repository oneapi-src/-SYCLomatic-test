// ====------ dnnl_utils_dropout.cpp --------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
// test_feature:engine_ext
// test_feature:memory_desc_ext
// test_feature:dropout_desc
// test_feature:dropout_forward
// test_feature:dropout_backward
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dnnl_utils.hpp>
#include <iostream>

void test1() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;

    handle.create_engine();

    int n = 4, c = 3, h = 32, w = 32;
    int ele_num = n * c * h * w;
    float *data, *out, *d_data, *d_out;

    data = sycl::malloc_shared<float>(ele_num, q_ct1);
    out = sycl::malloc_shared<float>(ele_num, q_ct1);
    d_data = sycl::malloc_shared<float>(ele_num, q_ct1);
    d_out = sycl::malloc_shared<float>(ele_num, q_ct1);

    for(int i = 0; i < ele_num; i++){
      data[i] = 2.f;
      d_out[i] = 3.f;
    }

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, n, c, h, w);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, n, c, h, w);

    float dropout = 0.8f;

    size_t reserve_size;
    void *reserve;
    size_t state_size;
    void *state;
    state_size = handle.get_dropout_state_size();
    reserve_size =
        dpct::dnnl::engine_ext::get_dropout_workspace_size(dataTensor);
    reserve = (void *)sycl::malloc_device(reserve_size, q_ct1);
    state = (void *)sycl::malloc_device(state_size, q_ct1);

    dpct::dnnl::dropout_desc desc;
    desc.init();
    desc.set(handle, dropout, state, state_size, 1231);
    handle.async_dropout_forward(desc, dataTensor, data, outTensor, out,
                                 reserve, reserve_size);

    dev_ct1.queues_wait_and_throw();
    float sum = 0.f, ave = 0.f, expect = 0.f, precision = 1.e-1;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }

    expect = 2.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }
    handle.async_dropout_backward(desc, dataTensor, d_out, outTensor, d_data,
                                  reserve, reserve_size);
    dev_ct1.queues_wait_and_throw();
    sum = 0.f;
    for(int i = 0; i < ele_num; i++){
      sum += d_data[i];
    }
    expect = 3.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "test failed" << std::endl;
        exit(-1);
    }
    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
    sycl::free(d_data, q_ct1);
    sycl::free(d_out, q_ct1);
}
void test2() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;

    handle.create_engine();

    int n = 4, c = 3, h = 32, w = 32;
    int ele_num = n * c * h * w;
    float *data, *out;

    data = sycl::malloc_shared<float>(ele_num, q_ct1);
    out = sycl::malloc_shared<float>(ele_num, q_ct1);

    for(int i = 0; i < ele_num; i++){
      data[i] = 2.f;
    }

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, n, c, h, w);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, n, c, h, w);

    float dropout = 0.8f;

    size_t reserve_size;
    void *reserve;
    size_t state_size;
    void *state;
    state_size = handle.get_dropout_state_size();
    reserve_size =
        dpct::dnnl::engine_ext::get_dropout_workspace_size(dataTensor);
    reserve = (void *)sycl::malloc_device(reserve_size, q_ct1);
    state = (void *)sycl::malloc_device(state_size, q_ct1);

    dpct::dnnl::dropout_desc desc;
    desc.init();
    desc.set(handle, dropout, state, state_size, 1231);
    handle.async_dropout_forward(desc, dataTensor, data, outTensor, out,
                                 reserve, reserve_size);

    dev_ct1.queues_wait_and_throw();
    float sum = 0.f, ave = 0.f, expect = 0.f, precision = 1.e-1;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }

    expect = 2.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }

    dpct::dnnl::dropout_desc desc2;
    desc2.init();
    desc2.restore(handle, dropout, state, state_size, 1231);
    handle.async_dropout_forward(desc2, dataTensor, data, outTensor, out,
                                 reserve, reserve_size);

    dev_ct1.queues_wait_and_throw();
    sum = 0.f;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}
int main() {
    test1();
    test2();
    std::cout << "test passed" << std::endl;
    return 0;
}
