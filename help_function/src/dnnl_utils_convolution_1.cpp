// ====------ dnnl_utils_convolution_1.cpp --------------===////
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
    dpct::dnnl::memory_desc_ext filterTensor;
    handle.create_engine();

                int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 4, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    filterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                     dpct::library_data_t::real_float, 4, filterdim);

    float *data, *out, *filter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
    }
    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    filter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1.memcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw)
        .wait();

    dpct::dnnl::convolution_desc covdes;
                covdes.set(0, 0, 1, 1, 1, 1);

    size_t size;
    void *workspacesize;
    size = 0;
    workspacesize = (void *)sycl::malloc_device(size, q_ct1);

    float alpha = 2.5f, beta = 1.5f;
    handle.async_convolution_forward(covdes, dnnl::algorithm::convolution_auto, alpha,
                               dataTensor, data, filterTensor, filter, beta,
                               outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow)
        .wait();

    std::vector<float> expect = {
        17260, 17561.5, 17863, 18164.5,
        18766, 19067.5, 19369, 19670.5,
        20272, 20573.5, 20875, 21176.5,
        21778, 22079.5, 22381, 22682.5,
        
        43204, 44145.5, 45087, 46028.5,
        47910, 48851.5, 49793, 50734.5,
        52616, 53557.5, 54499, 55440.5,
        57322, 58263.5, 59205, 60146.5,
        
        69148, 70729.5, 72311, 73892.5,
        77054, 78635.5, 80217, 81798.5,
        84960, 86541.5, 88123, 89704.5,
        92866, 94447.5, 96029, 97610.5,
        
        95092, 97313.5, 99535, 101756,
        106198, 108420, 110641, 112862,
        117304, 119526, 121747, 123968,
        128410, 130632, 132853, 135074,

        47356, 47657.5, 47959, 48260.5,
        48862, 49163.5, 49465, 49766.5,
        50368, 50669.5, 50971, 51272.5,
        51874, 52175.5, 52477, 52778.5,
        
        137300, 138242, 139183, 140124,
        142006, 142948, 143889, 144830,
        146712, 147654, 148595, 149536,
        151418, 152360, 153301, 154242,
        
        227244, 228826, 230407, 231988,
        235150, 236732, 238313, 239894,
        243056, 244638, 246219, 247800,
        250962, 252544, 254125, 255706,
        
        317188, 319410, 321631, 323852,
        328294, 330516, 332737, 334958,
        339400, 341622, 343843, 346064,
        350506, 352728, 354949, 357170,               
        };
    check(expect, host_out, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    dpct::dnnl::memory_desc_ext filterTensor;
    handle.create_engine();

                int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    filterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                     dpct::library_data_t::real_float, 4, filterdim);

    float *data, *out, *filter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
    }
    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    filter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1.memcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw)
        .wait();

    dpct::dnnl::convolution_desc covdes;
                covdes.set(0, 0, 1, 1, 1, 1);
    covdes.set_group_count(2);
    size_t size;
    void *workspacesize;
    size = 0;
    workspacesize = (void *)sycl::malloc_device(size, q_ct1);

    float alpha = 2.5f, beta = 1.5f;
    handle.async_convolution_forward(covdes, dnnl::algorithm::convolution_auto, alpha,
                               dataTensor, data, filterTensor, filter, beta,
                               outTensor, out);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow)
        .wait();

    std::vector<float> expect = {
        1640, 1711.5, 1783, 1854.5,
        1996, 2067.5, 2139, 2210.5,
        2352, 2423.5, 2495, 2566.5,
        2708, 2779.5, 2851, 2922.5,
        
        4144, 4375.5, 4607, 4838.5,
        5300, 5531.5, 5763, 5994.5,
        6456, 6687.5, 6919, 7150.5,
        7612, 7843.5, 8075, 8306.5,
        
        26148, 26539.5, 26931, 27322.5,
        28104, 28495.5, 28887, 29278.5,
        30060, 30451.5, 30843, 31234.5,
        32016, 32407.5, 32799, 33190.5,
        
        36652, 37203.5, 37755, 38306.5,
        39408, 39959.5, 40511, 41062.5,
        42164, 42715.5, 43267, 43818.5,
        44920, 45471.5, 46023, 46574.5,

        8736, 8807.5, 8879, 8950.5,
        9092, 9163.5, 9235, 9306.5,
        9448, 9519.5, 9591, 9662.5,
        9804, 9875.5, 9947, 10018.5,
        
        27240, 27471.5, 27703, 27934.5,
        28396, 28627.5, 28859, 29090.5,
        29552, 29783.5, 30015, 30246.5,
        30708, 30939.5, 31171, 31402.5,
        
        65244, 65635.5, 66027, 66418.5,
        67200, 67591.5, 67983, 68374.5,
        69156, 69547.5, 69939, 70330.5,
        71112, 71503.5, 71895, 72286.5,
        
        91748, 92299.5, 92851, 93402.5,
        94504, 95055.5, 95607, 96158.5,
        97260, 97811.5, 98363, 98914.5,
        100016, 100568, 101119, 101670,               
        };
    check(expect, host_out, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
