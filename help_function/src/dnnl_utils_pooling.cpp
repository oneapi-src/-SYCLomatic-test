// ====------ dnnl_utils_pooling.cpp --------------===//
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
// test_feature:pooling_desc
// test_feature:pooling_forward
// test_feature:pooling_backward
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
void test1() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;

    handle.create_engine();

    sycl::queue *stream1;
    stream1 = dev_ct1.create_queue();
    handle.set_queue(stream1);

    dpct::dnnl::pooling_desc desc;

    desc.set(dnnl::algorithm::pooling_max, 4, 4, 3, 3, 2, 2);

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    int on, oc, oh, ow;
    desc.get_forward_output_dim(dataTensor, &on, &oc, &oh, &ow);

    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    int ele_num2 = on * oc * oh * ow;
    std::vector<HT> host_out(ele_num2);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
    }

    for(int i = 0; i < ele_num2; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num2 * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num2 * sizeof(HT)).wait();

    float alpha = 1.f, beta = 0.f;

    auto s = (handle.async_pooling_forward(desc, alpha, dataTensor, data, beta,
                                     outTensor, out),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num2 * sizeof(HT)).wait();

    std::vector<float> expect = {
        0, 2, 4, 4,
        10, 12, 14, 14,
        20, 22, 24, 24,
        20, 22, 24, 24,
        25, 27, 29, 29,
        35, 37, 39, 39,
        45, 47, 49, 49,
        45, 47, 49, 49
      };
      check(expect, host_out, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, diffdataTensor,
        diffoutTensor;

    handle.create_engine();

    sycl::queue *stream1;
    stream1 = dev_ct1.create_queue();
    handle.set_queue(stream1);

    dpct::dnnl::pooling_desc desc;

    desc.set(dnnl::algorithm::pooling_max, 4, 4, 3, 3, 2, 2);

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

    diffdataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

    int on, oc, oh, ow;
    desc.get_forward_output_dim(dataTensor, &on, &oc, &oh, &ow);

    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);
    diffoutTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);
    int ele_num2 = on * oc * oh * ow;

    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num2);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num2);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < ele_num2; i++) {
        host_out[i] = i;
        host_diffout[i] = 1.f;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num2 * sizeof(HT), *stream1);
    diffdata = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    diffout = (HT *)sycl::malloc_device(ele_num2 * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num2 * sizeof(HT)).wait();
    stream1->memcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(diffout, host_diffout.data(), ele_num2 * sizeof(HT)).wait();

    dnnl::memory pooling_workspace;

    float alpha = 1.5f, beta = 1.f;
    handle.async_pooling_forward(desc, alpha, dataTensor, data, beta, outTensor, out, &pooling_workspace);
    dev_ct1.queues_wait_and_throw();    
    stream1->memcpy(host_out.data(), out, ele_num2 * sizeof(HT)).wait();
    alpha = 1.5f, beta = 1.f;

    auto s = (handle.async_pooling_backward(desc, alpha, outTensor, out,
                                      diffoutTensor, diffout, dataTensor, data,
                                      beta, diffdataTensor, diffdata, &pooling_workspace),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        1.5, 1, 3.5, 3, 7,
        5, 6, 7, 8, 9,
        11.5, 11, 13.5, 13, 17,
        15, 16, 17, 18, 19,
        23, 21, 25, 23, 30,
        26.5, 26, 28.5, 28, 32,
        30, 31, 32, 33, 34,
        36.5, 36, 38.5, 38, 42,
        40, 41, 42, 43, 44,
        48, 46, 50, 48, 55
      };
    check(expect, host_diffdata, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
    sycl::free(diffdata, *stream1);
    sycl::free(diffout, *stream1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}