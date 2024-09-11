// ====------ dnnl_utils_softmax.cpp --------------===//
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
// test_feature:softmax_mode
// test_feature:softmax_algorithm
// test_feature:softmax_forward
// test_feature:softmax_backward
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
    std::cout << "test1" << std::endl;
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
        host_data[i] = 10 * i;
        host_out[i] = i;
    }

    data = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);
    out = (HT *)sycl::malloc_device(ele_num * sizeof(HT), *stream1);

    stream1->memcpy(data, host_data.data(), ele_num * sizeof(HT)).wait();
    stream1->memcpy(out, host_out.data(), ele_num * sizeof(HT)).wait();

    float alpha = 2.f, beta = 1.5f;

    auto s = (handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::normal,
                                     dpct::dnnl::softmax_mode::channel, alpha,
                                     dataTensor, data, beta, outTensor, out),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        0, 1.5, 3, 4.5, 6,
        7.5, 9, 10.5, 12, 13.5,
        15, 16.5, 18, 19.5, 21,
        22.5, 24, 25.5, 27, 28.5,
        30, 31.5, 33, 34.5, 36,
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
    std::cout << "test2" << std::endl;
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

    float alpha = 1.5f, beta = 0.f;
    handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::normal,
                           dpct::dnnl::softmax_mode::channel, alpha, dataTensor,
                           data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    alpha = 2.f, beta = 0.f;

    auto s = (handle.async_softmax_backward(dpct::dnnl::softmax_algorithm::normal,
                                      dpct::dnnl::softmax_mode::channel, alpha,
                                      outTensor, out, diffoutTensor, diffout,
                                      beta, diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621
      };
    check(expect, host_diffdata, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
    sycl::free(diffdata, *stream1);
    sycl::free(diffout, *stream1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test3() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    std::cout << "test3" << std::endl;
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, diffdataTensor,
        diffoutTensor;

    handle.create_engine();

    sycl::queue *stream1;
    stream1 = dev_ct1.create_queue();
    handle.set_queue(stream1);

    int n = 2, c = 2, h = 5, w = 5;
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

    float alpha = 1.5f, beta = 0.f;
    handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::normal,
                           dpct::dnnl::softmax_mode::instance, alpha,
                           dataTensor, data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    alpha = 2.f, beta = 0.f;

    auto s = (handle.async_softmax_backward(dpct::dnnl::softmax_algorithm::normal,
                                      dpct::dnnl::softmax_mode::instance, alpha,
                                      outTensor, out, diffoutTensor, diffout,
                                      beta, diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        -0.00107016, -0.00118271, -0.0013071, -0.00144457, -0.0015965,
        -0.0017644, -0.00194997, -0.00215505, -0.0023817, -0.00263218,
        -0.00290901, -0.00321495, -0.00355307, -0.00392675, -0.00433973,
        -0.00479614, -0.00530056, -0.00585802, -0.00647412, -0.00715501,
        -0.0079075, -0.00873915, -0.00965825, -0.010674, -0.0117966,
        -0.0130373, -0.0144084, -0.0159238, -0.0175985, -0.0194493,
        -0.0214948, -0.0237555, -0.0262538, -0.029015, -0.0320665,
        -0.035439, -0.0391661, -0.0432853, -0.0478376, -0.0528688,
        -0.058429, -0.064574, -0.0713654, -0.0788709, -0.0871658,
        -0.0963331, -0.106465, -0.117662, -0.130036, -0.143712,
        -0.00107016, -0.00118271, -0.0013071, -0.00144457, -0.0015965,
        -0.0017644, -0.00194997, -0.00215505, -0.0023817, -0.00263218,
        -0.00290901, -0.00321495, -0.00355307, -0.00392675, -0.00433973,
        -0.00479615, -0.00530056, -0.00585803, -0.00647412, -0.00715501,
        -0.00790751, -0.00873915, -0.00965825, -0.010674, -0.0117966,
        -0.0130373, -0.0144084, -0.0159238, -0.0175985, -0.0194493,
        -0.0214948, -0.0237555, -0.0262538, -0.029015, -0.0320665,
        -0.035439, -0.0391661, -0.0432853, -0.0478376, -0.0528688,
        -0.058429, -0.0645741, -0.0713653, -0.0788709, -0.0871659,
        -0.0963332, -0.106465, -0.117662, -0.130036, -0.143712
      };
    check(expect, host_diffdata, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
    sycl::free(diffdata, *stream1);
    sycl::free(diffout, *stream1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test4() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    std::cout << "test4" << std::endl;
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

    float alpha = 1.5f, beta = 0.f;
    handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::log,
                           dpct::dnnl::softmax_mode::channel, alpha, dataTensor,
                           data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    alpha = 2.f, beta = 3.f;

    auto s = (handle.async_softmax_backward(dpct::dnnl::softmax_algorithm::log,
                                      dpct::dnnl::softmax_mode::channel, alpha,
                                      outTensor, out, diffoutTensor, diffout,
                                      beta, diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        1.91643, 4.91643, 7.91643, 10.9164, 13.9164,
        16.9164, 19.9164, 22.9164, 25.9164, 28.9164,
        31.9164, 34.9164, 37.9164, 40.9164, 43.9164,
        46.9164, 49.9164, 52.9164, 55.9164, 58.9164,
        61.9164, 64.9164, 67.9164, 70.9164, 73.9164,
        73.4464, 76.4464, 79.4464, 82.4464, 85.4464,
        88.4464, 91.4464, 94.4464, 97.4464, 100.446,
        103.446, 106.446, 109.446, 112.446, 115.446,
        118.446, 121.446, 124.446, 127.446, 130.446,
        133.446, 136.446, 139.446, 142.446, 145.446
      };
    check(expect, host_diffdata, expect.size(), 1e-3);

    sycl::free(data, *stream1);
    sycl::free(out, *stream1);
    sycl::free(diffdata, *stream1);
    sycl::free(diffout, *stream1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test5() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    std::cout << "test5" << std::endl;
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

    float alpha = 1.5f, beta = 0.f;
    handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::normal,
                           dpct::dnnl::softmax_mode::channel, alpha, dataTensor,
                           data, beta, outTensor, out);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    alpha = 2.f, beta = 1.5f;

    auto s = (handle.async_softmax_backward(dpct::dnnl::softmax_algorithm::normal,
                                      dpct::dnnl::softmax_mode::channel, alpha,
                                      outTensor, out, diffoutTensor, diffout,
                                      beta, diffdataTensor, diffdata),
              0);
    dev_ct1.queues_wait_and_throw();
    stream1->memcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT)).wait();

    std::vector<float> expect = {
        -0.113787, 1.38621, 2.88621, 4.38621, 5.88621,
        7.38621, 8.88621, 10.3862, 11.8862, 13.3862,
        14.8862, 16.3862, 17.8862, 19.3862, 20.8862,
        22.3862, 23.8862, 25.3862, 26.8862, 28.3862,
        29.8862, 31.3862, 32.8862, 34.3862, 35.8862,
        36.1138, 37.6138, 39.1138, 40.6138, 42.1138,
        43.6138, 45.1138, 46.6138, 48.1138, 49.6138,
        51.1138, 52.6138, 54.1138, 55.6138, 57.1138,
        58.6138, 60.1138, 61.6138, 63.1138, 64.6138,
        66.1138, 67.6138, 69.1138, 70.6138, 72.1138
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
    test3<dpct::library_data_t::real_float>();
    test4<dpct::library_data_t::real_float>();
    test5<dpct::library_data_t::real_float>();

    std::cout << "test passed" << std::endl;
    return 0;
}