// ====------ dnnl_utils_reduction.cpp --------------===////
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
// test_feature:reduction_op
// test_feature:reduction

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

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 2, oh = 6, ow = 1;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i - 25.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    float alpha = 2.5f, beta = 1.5f;

    dpct::dnnl::reduction_op reducedesc;

    reducedesc = dpct::dnnl::reduction_op::sum;
    size_t ws_size;
    ws_size = 0;
    void * ws;
    ws = (void *)sycl::malloc_device(ws_size, q_ct1);

    handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {-337.5, -246, -154.5, -63,  28.5,   120,
        211.5,  303,  394.5,  486,  577.5,  669,
        760.5,  852,  943.5,  1035, 1126.5, 1218,
        1309.5, 1401, 1492.5, 1584, 1675.5, 1767};
    check(expect, host_out, expect.size(), 1e-3);

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

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i - 25.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    float alpha = 2.5f, beta = 1.5f;

    dpct::dnnl::reduction_op reducedesc;

    reducedesc = dpct::dnnl::reduction_op::sum;
    size_t ws_size;
    ws_size = 0;
    void * ws;
    ws = (void *)sycl::malloc_device(ws_size, q_ct1);

    handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        -135, 46.5,   228,  409.5,  591,  772.5,
        2034, 2215.5, 2397, 2578.5, 2760, 2941.5};
    check(expect, host_out, expect.size(), 1e-3);

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

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i - 25.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    float alpha = 2.5f, beta = 1.5f;

    dpct::dnnl::reduction_op reducedesc;

    reducedesc = dpct::dnnl::reduction_op::norm1;
    size_t ws_size;
    ws_size = 0;
    void * ws;
    ws = (void *)sycl::malloc_device(ws_size, q_ct1);

    handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        540,  541.5,  543,  544.5,  596,  772.5,
        2034, 2215.5, 2397, 2578.5, 2760, 2941.5};
    check(expect, host_out, expect.size(), 1e-3);

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

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i - 25.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    float alpha = 2.5f, beta = 1.5f;

    dpct::dnnl::reduction_op reducedesc;

    reducedesc = dpct::dnnl::reduction_op::norm2;
    size_t ws_size;
    ws_size = 0;
    void * ws;
    ws = (void *)sycl::malloc_device(ws_size, q_ct1);


    handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        161.361, 158.623, 172.521, 199.916,
        236.299, 278.217, 614.176, 666.005,
        718.072, 770.327, 822.736, 875.271};
    check(expect, host_out, expect.size(), 1e-3);

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



    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = (i - 60.f) / 50;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(sizeof(HT) * in * ic * ih * iw, q_ct1);
    out = (HT *)sycl::malloc_device(sizeof(HT) * on * oc * oh * ow, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow).wait();

    float alpha = 2.5f, beta = 1.5f;

    dpct::dnnl::reduction_op reducedesc;

    reducedesc = dpct::dnnl::reduction_op::mul_no_zeros;
    size_t ws_size;
    ws_size = 0;
    void * ws;
    ws = (void *)sycl::malloc_device(ws_size, q_ct1);


    handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        0.0357702, 1.50255, 3.00006, 4.5,
        6,         7.5,     9.00151, 10.5241,
        12.2083,   14.734,  20.6591, 38.0143};
    check(expect, host_out, expect.size(), 1e-3);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    test3<dpct::library_data_t::real_float>();
    test4<dpct::library_data_t::real_float>();
    test5<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
