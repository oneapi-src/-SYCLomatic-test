// ====------ dnnl_utils_convolution_3.cpp --------------===////
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
// test_feature:convolution_backward_data

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
    dpct::dnnl::memory_desc_ext diffdataTensor, diffoutTensor;
    dpct::dnnl::memory_desc_ext filterTensor, difffilterTensor;
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
    diffdataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                       dpct::library_data_t::real_float, in, ic, ih, iw);
    diffoutTensor.set(dpct::dnnl::memory_format_tag::nchw,
                      dpct::library_data_t::real_float, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    filterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                     dpct::library_data_t::real_float, 4, filterdim);
    difffilterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                         dpct::library_data_t::real_float, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);
    std::vector<float> host_diffdata(in * ic * ih * iw, 1.0f);
    std::vector<float> host_diffout(on * oc * oh * ow, 0.0f);
    std::vector<float> host_difffilter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_diffout[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
        host_difffilter[i] = i;
    }

    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    filter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);
    diffdata =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    diffout =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    difffilter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1.memcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw)
        .wait();
    q_ct1
        .memcpy(diffdata, host_diffdata.data(),
                sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1
        .memcpy(diffout, host_diffout.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1
        .memcpy(difffilter, host_difffilter.data(),
                sizeof(float) * fk * fc * fh * fw)
        .wait();

    dpct::dnnl::convolution_desc covdes;
                covdes.set(0, 0, 1, 1, 1, 1);

    size_t size;
    void *workspacesize;
    size = 0;
    workspacesize = (void *)sycl::malloc_device(size, q_ct1);

    float alpha = 2.5f, beta = 1.5f;
    handle.async_convolution_backward_data(
        covdes, dnnl::algorithm::convolution_direct, alpha, filterTensor,
        filter, diffoutTensor, diffout, beta, diffdataTensor, diffdata);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num)
        .wait();

    std::vector<float> expect = {
        8960, 18401.5, 18893, 19384.5, 9956,
        19367.5, 39749, 40770.5, 41792, 21453.5,
        21375, 43836.5, 44858, 45879.5, 23541,
        23382.5, 47924, 48945.5, 49967, 25628.5,
        12590, 25771.5, 26303, 26834.5, 13766,
        
        9957.5, 20399, 20970.5, 21542, 11073.5,
        21485, 44026.5, 45208, 46389.5, 23811,
        23812.5, 48754, 49935.5, 51117, 26218.5,
        26140, 53481.5, 54663, 55844.5, 28626,
        14067.5, 28729, 29340.5, 29952, 15363.5,
        
        10955, 22396.5, 23048, 23699.5, 12191,
        23602.5, 48304, 49645.5, 50987, 26168.5,
        26250, 53671.5, 55013, 56354.5, 28896,
        28897.5, 59039, 60380.5, 61722, 31623.5,
        15545, 31686.5, 32378, 33069.5, 16961,
        
        11952.5, 24394, 25125.5, 25857, 13308.5,
        25720, 52581.5, 54083, 55584.5, 28526,
        28687.5, 58589, 60090.5, 61592, 31573.5,
        31655, 64596.5, 66098, 67599.5, 34621,
        17022.5, 34644, 35415.5, 36187, 18558.5,

        24470, 49911.5, 50403, 50894.5, 26106,
        51517.5, 105179, 106200, 107222, 54883.5,
        53525, 109266, 110288, 111310, 56971,
        55532.5, 113354, 114376, 115397, 59058.5,
        29380, 59841.5, 60373, 60904.5, 31196,
        
        28027.5, 57029, 57600.5, 58172, 29783.5,
        58755, 119696, 120878, 122060, 62361,
        61082.5, 124424, 125606, 126787, 64768.5,
        63410, 129152, 130333, 131514, 67176,
        33417.5, 67919, 68530.5, 69142, 35353.5,
        
        31585, 64146.5, 64798, 65449.5, 33461,
        65992.5, 134214, 135556, 136897, 69838.5,
        68640, 139582, 140923, 142264, 72566,
        71287.5, 144949, 146290, 147632, 75293.5,
        37455, 75996.5, 76688, 77379.5, 39511,
        
        35142.5, 71264, 71995.5, 72727, 37138.5,
        73230, 148732, 150233, 151734, 77316,
        76197.5, 154739, 156240, 157742, 80363.5,
        79165, 160746, 162248, 163750, 83411,
        41492.5, 84074, 84845.5, 85617, 43668.5,        
        };
    check(expect, host_diffdata, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor;
    dpct::dnnl::memory_desc_ext diffdataTensor, diffoutTensor;
    dpct::dnnl::memory_desc_ext filterTensor, difffilterTensor;
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
    diffdataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                       dpct::library_data_t::real_float, in, ic, ih, iw);
    diffoutTensor.set(dpct::dnnl::memory_format_tag::nchw,
                      dpct::library_data_t::real_float, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    filterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                     dpct::library_data_t::real_float, 4, filterdim);
    difffilterTensor.set(dpct::dnnl::memory_format_tag::nchw,
                         dpct::library_data_t::real_float, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);
    std::vector<float> host_diffdata(in * ic * ih * iw, 1.0f);
    std::vector<float> host_diffout(on * oc * oh * ow, 0.0f);
    std::vector<float> host_difffilter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_diffout[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
        host_difffilter[i] = i;
    }

    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    filter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);
    diffdata =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    diffout =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    difffilter =
        (float *)sycl::malloc_device(sizeof(float) * fk * fc * fh * fw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1.memcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw)
        .wait();
    q_ct1
        .memcpy(diffdata, host_diffdata.data(),
                sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1
        .memcpy(diffout, host_diffout.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1
        .memcpy(difffilter, host_difffilter.data(),
                sizeof(float) * fk * fc * fh * fw)
        .wait();

    dpct::dnnl::convolution_desc covdes;
                covdes.set(0, 0, 1, 1, 1, 1);
    covdes.set_group_count(2);

    size_t size;
    void *workspacesize;
    size = 0;
    workspacesize = (void *)sycl::malloc_device(size, q_ct1);

    float alpha = 2.5f, beta = 1.5f;
    handle.async_convolution_backward_data(
        covdes, dnnl::algorithm::convolution_direct, alpha, filterTensor,
        filter, diffoutTensor, diffout, beta, diffdataTensor, diffdata);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num)
        .wait();

    std::vector<float> expect = {
        320, 701.5, 748, 794.5, 441,
        807.5, 1759, 1870.5, 1982, 1093.5,
        1015, 2206.5, 2318, 2429.5, 1341,
        1222.5, 2654, 2765.5, 2877, 1588.5,
        790, 1681.5, 1748, 1814.5, 1001,
        
        517.5, 1079, 1165.5, 1252, 698.5,
        1245, 2636.5, 2828, 3019.5, 1651,
        1612.5, 3404, 3595.5, 3787, 2058.5,
        1980, 4171.5, 4363, 4554.5, 2466,
        1227.5, 2539, 2645.5, 2752, 1498.5,
        
        4235, 8696.5, 8903, 9109.5, 4756,
        9202.5, 18954, 19385.5, 19817, 10288.5,
        10050, 20681.5, 21113, 21544.5, 11176,
        10897.5, 22409, 22840.5, 23272, 12063.5,
        5985, 12236.5, 12463, 12689.5, 6596,
        
        5072.5, 10354, 10600.5, 10847, 5653.5,
        10920, 22391.5, 22903, 23414.5, 12126,
        11927.5, 24439, 24950.5, 25462, 13173.5,
        12935, 26486.5, 26998, 27509.5, 14221,
        7062.5, 14374, 14640.5, 14907, 7733.5,

        1750, 3731.5, 3778, 3824.5, 2191,
        4157.5, 8949, 9060.5, 9172, 5083.5,
        4365, 9396.5, 9508, 9619.5, 5331,
        4572.5, 9844, 9955.5, 10067, 5578.5,
        2860, 5991.5, 6058, 6124.5, 3391,
        
        3227.5, 6669, 6755.5, 6842, 3728.5,
        7155, 14946.5, 15138, 15329.5, 8201,
        7522.5, 15714, 15905.5, 16097, 8608.5,
        7890, 16481.5, 16673, 16864.5, 9016,
        4577.5, 9409, 9515.5, 9622, 5168.5,
        
        10785, 21966.5, 22173, 22379.5, 11626,
        22792.5, 46624, 47055.5, 47487, 24518.5,
        23640, 48351.5, 48783, 49214.5, 25406,
        24487.5, 50079, 50510.5, 50942, 26293.5,
        13175, 26786.5, 27013, 27239.5, 14106,
        
        12902.5, 26184, 26430.5, 26677, 13803.5,
        27070, 55181.5, 55693, 56204.5, 28916,
        28077.5, 57229, 57740.5, 58252, 29963.5,
        29085, 59276.5, 59788, 60299.5, 31011,
        15532.5, 31484, 31750.5, 32017, 16523.5,
        };
    check(expect, host_diffdata, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
