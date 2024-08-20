// ====------ dnnl_utils_convolution_5.cpp --------------===////
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
// test_feature:memory_desc_ext
// test_feature:convolution_desc
// test_feature:convolution_forward
// test_feature:convolution_forward_ex
// test_feature:convolution_backward_bias

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
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, biasTensor;
    dpct::dnnl::memory_desc_ext filterTensor;
        auto status = (handle.create_engine(), 0);

                    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 4, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    std::vector<int> bias_dim = {1, oc, 1, 1};
    std::vector<int> bias_stride = {oc, 1, 1, 1};
    int bele_num = oc * 1;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);
    biasTensor.set(dpct::library_data_t::real_float, 4, bias_dim.data(),
                   bias_stride.data());
    int filterdim[4] = {fk, fc, fh, fw};
    filterTensor.set(dpct::dnnl::memory_format_tag::nhwc,
                     dpct::library_data_t::real_float, 4, filterdim);

    float *data, *out, *filter, *z, *bias;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_z(oele_num, 0.0f);
    std::vector<float> host_bias(bele_num, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_z[i] = i;
    }
    for(int i = 0; i < bele_num; i++) {
        host_bias[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
    }

    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    z = (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    bias = sycl::malloc_device<float>(bele_num, q_ct1);
    filter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * on * oc * oh * ow).wait();
    q_ct1.memcpy(bias, host_bias.data(), sizeof(float) * bele_num).wait();

    float alpha = 2.5f, beta = 1.5f;

    handle.async_convolution_backward_bias(alpha, outTensor, out, beta, biasTensor,
                                     bias);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_bias.data(), bias, sizeof(float) * bele_num).wait();

    std::vector<float> expect = {
        3160,

        4441.5,

        5723,

        7004.5 
        };
    check(expect, host_bias, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
