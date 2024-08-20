// ====------ dnnl_utils_lrn.cpp --------------===//
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
// test_feature:lrn_desc
// test_feature:lrn_forward
// test_feature:lrn_backward
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

    unsigned int local_size = 3;
    float lrn_alpha = 1.5f;
    float lrn_beta = 1.5f;
    float lrn_k = 1.f;

    dpct::dnnl::lrn_desc desc;

    desc.set(local_size, lrn_alpha, lrn_beta, lrn_k);

    float alpha = 2.f, beta = 1.5f;

    auto s = (handle.async_lrn_forward(desc, alpha, dataTensor, data, beta, outTensor,
                                 out),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        0, 1.50032, 3.00057, 4.50076, 6.0009,
        7.501, 9.00107, 10.5011, 12.0012, 13.5012,
        15.0012, 16.5012, 18.0012, 19.5011, 21.0011,
        22.5011, 24.0011, 25.501, 27.001, 28.501,
        30.0009, 31.5009, 33.0009, 34.5009, 36.0008,
        37.509, 39.0083, 40.5077, 42.0071, 43.5065,
        45.006, 46.5056, 48.0051, 49.5048, 51.0044,
        52.5041, 54.0038, 55.5035, 57.0033, 58.5031,
        60.0029, 61.5027, 63.0026, 64.5024, 66.0023,
        67.5021, 69.002, 70.5019, 72.0018, 73.5017
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
        host_data[i] = i;
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

    unsigned int local_size = 3;
    float lrn_alpha = 1.5f;
    float lrn_beta = 1.5f;
    float lrn_k = 1.f;

    dpct::dnnl::lrn_desc desc;

    desc.set(local_size, lrn_alpha, lrn_beta, lrn_k);

    float alpha = 1.5f, beta = 0.f;
    handle.async_lrn_forward(desc, alpha, dataTensor, data, beta, outTensor, out);

    alpha = 2000.f, beta = 0.f;
    dev_ct1.queues_wait_and_throw();

    auto s = (handle.async_lrn_backward(desc, alpha, outTensor, out, diffoutTensor,
                                  diffout, dataTensor, data, beta,
                                  diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();

    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();


    std::vector<float> expect = {
        0.360308, 0.28158, 0.21668, 0.163798, 0.121108,
        0.0869165, 0.059718, 0.0382204, 0.021336, 0.00816559,
        -0.00202841, -0.00984461, -0.0157668, -0.0201844, -0.0234096,
        -0.0256924, -0.0272326, -0.02819, -0.0286917, -0.0288397,
        -0.0287147, -0.0283811, -0.0278906, -0.0272837, -0.0265927,
        -0.717169, -0.67193, -0.623392, -0.574243, -0.526284,
        -0.480635, -0.437933, -0.398476, -0.362341, -0.329454,
        -0.299655, -0.272737, -0.248469, -0.226616, -0.206946,
        -0.189243, -0.173305, -0.158945, -0.145997, -0.13431,
        -0.123748, -0.114191, -0.105531, -0.0976739, -0.0905342
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