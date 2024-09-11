// ====------ dnnl_utils_binary.cpp --------------===////
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
// test_feature:binary_op
// test_feature:binary

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

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::add;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        12.5,17.25,22,26.75,31.5,
        36.25,41,45.75,50.5,55.25,
        60,64.75,69.5,74.25,79,
        83.75,88.5,93.25,98,102.75,
        107.5,112.25,117,121.75,126.5,

        131.25,136,140.75,145.5,150.25,
        155,159.75,164.5,169.25,174,
        178.75,183.5,188.25,193,197.75,
        202.5,207.25,212,216.75,221.5,
        226.25,231,235.75,240.5,245.25,

        250,254.75,259.5,264.25,269,
        273.75,278.5,283.25,288,292.75,
        297.5,302.25,307,311.75,316.5,
        321.25,326,330.75,335.5,340.25,
        345,349.75,354.5,359.25,364,

        368.75,373.5,378.25,383,387.75,
        392.5,397.25,402,406.75,411.5,
        416.25,421,425.75,430.5,435.25,
        440,444.75,449.5,454.25,459,
        463.75,468.5,473.25,478,482.75      
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    handle.create_engine();

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::mul;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        0,22.625,49,79.125,113,
        150.625,192,237.125,286,338.625,
        395,455.125,519,586.625,658,
        733.125,812,894.625,981,1071.12,
        1165,1262.62,1364,1469.12,1578,

        1690.62,1807,1927.12,2051,2178.62,
        2310,2445.12,2584,2726.62,2873,
        3023.12,3177,3334.62,3496,3661.12,
        3830,4002.62,4179,4359.12,4543,
        4730.62,4922,5117.12,5316,5518.62,

        5725,5935.12,6149,6366.62,6588,
        6813.12,7042,7274.62,7511,7751.12,
        7995,8242.62,8494,8749.12,9008,
        9270.62,9537,9807.12,10081,10358.6,
        10640,10925.1,11214,11506.6,11803,

        12103.1,12407,12714.6,13026,13341.1,
        13660,13982.6,14309,14639.1,14973,
        15310.6,15652,15997.1,16346,16698.6,
        17055,17415.1,17779,18146.6,18518,
        18893.1,19272,19654.6,20041,20431.1
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}
template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test3() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    handle.create_engine();


    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::min;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        0,3.5,7,10.5,14,
        17.5,21,24.5,28,31.5,
        35,38.5,42,45.5,49,
        52.5,56,59.5,63,66.5,
        70,73.5,77,80.5,84,

        87.5,91,94.5,98,101.5,
        105,108.5,112,115.5,119,
        122.5,126,129.5,133,136.5,
        140,143.5,147,150.5,154,
        157.5,161,164.5,168,171.5,

        175,178.25,181.5,184.75,188,
        191.25,194.5,197.75,201,204.25,
        207.5,210.75,214,217.25,220.5,
        223.75,227,230.25,233.5,236.75,
        240,243.25,246.5,249.75,253,

        256.25,259.5,262.75,266,269.25,
        272.5,275.75,279,282.25,285.5,
        288.75,292,295.25,298.5,301.75,
        305,308.25,311.5,314.75,318,
        321.25,324.5,327.75,331,334.25
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}
template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test4() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    handle.create_engine();


    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::max;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        12.5,15.75,19,22.25,25.5,
        28.75,32,35.25,38.5,41.75,
        45,48.25,51.5,54.75,58,
        61.25,64.5,67.75,71,74.25,
        77.5,80.75,84,87.25,90.5,

        93.75,97,100.25,103.5,106.75,
        110,113.25,116.5,119.75,123,
        126.25,129.5,132.75,136,139.25,
        142.5,145.75,149,152.25,155.5,
        158.75,162,165.25,168.5,171.75,

        175,178.5,182,185.5,189,
        192.5,196,199.5,203,206.5,
        210,213.5,217,220.5,224,
        227.5,231,234.5,238,241.5,
        245,248.5,252,255.5,259,

        262.5,266,269.5,273,276.5,
        280,283.5,287,290.5,294,
        297.5,301,304.5,308,311.5,
        315,318.5,322,325.5,329,
        332.5,336,339.5,343,346.5
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}
template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test5() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    handle.create_engine();

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::sqrt;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        0,3.22474,5.73205,8.12132,10.4495,
        12.7386,15,17.2404,19.4641,21.6742,
        23.873,26.062,28.2426,30.4159,32.5826,
        34.7434,36.899,39.0498,41.1962,43.3385,
        45.4772,47.6125,49.7446,51.8737,54,

        56.1237,58.245,60.364,62.4807,64.5955,
        66.7082,68.8191,70.9282,73.0356,75.1414,
        77.2457,79.3485,81.4498,83.5498,85.6485,
        87.746,89.8422,91.9373,94.0312,96.124,
        98.2158,100.307,102.396,104.485,106.573,

        108.66,110.746,112.832,114.916,117,
        119.083,121.165,123.247,125.327,127.407,
        129.487,131.566,133.644,135.721,137.798,
        139.874,141.95,144.025,146.1,148.173,
        150.247,152.32,154.392,156.464,158.536,

        160.607,162.677,164.747,166.817,168.886,
        170.954,173.023,175.091,177.158,179.225,
        181.292,183.358,185.424,187.489,189.554,
        191.619,193.683,195.747,197.811,199.874,
        201.937,204,206.062,208.124,210.186
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}
template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test6() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    handle.create_engine();

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f + 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }
    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);
    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    dpct::dnnl::binary_op OpDesc;

    OpDesc = dpct::dnnl::binary_op::neg;

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;

    auto status = (handle.async_binary(OpDesc, alpha0, outTensor, out, alpha1,
                                 dataTensor, data, beta, outTensor, out),
                   0);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        1,1.5,2,2.5,3,
        3.5,4,4.5,5,5.5,
        6,6.5,7,7.5,8,
        8.5,9,9.5,10,10.5,
        11,11.5,12,12.5,13,

        13.5,14,14.5,15,15.5,
        16,16.5,17,17.5,18,
        18.5,19,19.5,20,20.5,
        21,21.5,22,22.5,23,
        23.5,24,24.5,25,25.5,

        26,26.5,27,27.5,28,
        28.5,29,29.5,30,30.5,
        31,31.5,32,32.5,33,
        33.5,34,34.5,35,35.5,
        36,36.5,37,37.5,38,

        38.5,39,39.5,40,40.5,
        41,41.5,42,42.5,43,
        43.5,44,44.5,45,45.5,
        46,46.5,47,47.5,48,
        48.5,49,49.5,50,50.5
        };
    check(expect, host_out, expect.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    test3<dpct::library_data_t::real_float>();
    test4<dpct::library_data_t::real_float>();
    test5<dpct::library_data_t::real_float>();
    test6<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
