// ====------ dnnl_utils_activation.cpp ---------------===//
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
// test_feature:activation_desc
// test_feature:activation_forward
// test_feature:activation_backward
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

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num * sizeof(HT)).wait();

    dpct::dnnl::activation_desc desc;

    desc.set(dnnl::algorithm::eltwise_logistic_use_dst_for_bwd, 0.f);

    float alpha = 2.f, beta = 1.5f;

    auto s = (handle.async_activation_forward(desc, alpha, dataTensor, data, beta,
                                        outTensor, out),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        1, 2.96212, 4.76159, 6.40515, 7.96403,
        9.48661, 10.9951, 12.4982, 13.9993, 15.4998,
        16.9999, 18.5, 20, 21.5, 23,
        24.5, 26, 27.5, 29, 30.5,
        32, 33.5, 35, 36.5, 38,
        39.5, 41, 42.5, 44, 45.5,
        47, 48.5, 50, 51.5, 53,
        54.5, 56, 57.5, 59, 60.5,
        62, 63.5, 65, 66.5, 68,
        69.5, 71, 72.5, 74, 75.5    
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

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    diffdataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    diffoutTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    diffdata = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    diffout = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(diffout, host_diffout.data(), ele_num * sizeof(HT)).wait();

    dpct::dnnl::activation_desc desc;

    desc.set(dnnl::algorithm::eltwise_logistic_use_dst_for_bwd, 0.f);

    float alpha = 1.5f, beta = 0.f;
    handle.async_activation_forward(desc, alpha, dataTensor, data, beta, outTensor,
                              out);

    alpha = 2.f, beta = 0.f;


    auto s = (handle.async_activation_backward(desc, alpha, outTensor, out,
                                         diffoutTensor, diffout, dataTensor,
                                         data, beta, diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        0.375, 0.334723, 0.289074, 0.238399, 0.183142,
        0.123828, 0.0610447, -0.00457374, -0.072368, -0.141673,
        -0.211834, -0.282226, -0.352262, -0.42141, -0.489194,
        -0.555202, -0.61909, -0.680577, -0.739441, -0.795526,
        -0.848724, -0.898978, -0.946273, -0.990628, -1.03209,
        -1.07075, -1.10668, -1.14001, -1.17084, -1.19932,
        -1.22557, -1.24972, -1.27191, -1.29227, -1.31092,
        -1.32799, -1.3436, -1.35786, -1.37087, -1.38273,
        -1.39354, -1.40338, -1.41234, -1.42049, -1.42789,
        -1.43462, -1.44073, -1.44629, -1.45132, -1.4559             
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