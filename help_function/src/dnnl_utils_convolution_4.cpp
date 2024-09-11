// ====------ dnnl_utils_convolution_4.cpp --------------===////
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
// test_feature:convolution_backward_weight

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

    float alpha = 1.5f, beta = 1.5f;
    handle.async_convolution_backward_weight(
        covdes, dnnl::algorithm::convolution_direct, alpha, dataTensor, data,
        diffoutTensor, diffout, beta, difffilterTensor, difffilter);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_difffilter.data(), difffilter, sizeof(float) * fele_num)
        .wait();

    std::vector<float> expect = {
        189924, 191822,
        199407, 201304,
        
        237330, 239228,
        246813, 248710,
        
        284736, 286634,
        294219, 296116,
        
        332142, 334040,
        341625, 343522,

        235260, 237926,
        248583, 251248,
        
        301866, 304532,
        315189, 317854,
        
        368472, 371138,
        381795, 384460,
        
        435078, 437744,
        448401, 451066,

        280596, 284030,
        297759, 301192,
        
        366402, 369836,
        383565, 386998,
        
        452208, 455642,
        469371, 472804,
        
        538014, 541448,
        555177, 558610,

        325932, 330134,
        346935, 351136,
        
        430938, 435140,
        451941, 456142,
        
        535944, 540146,
        556947, 561148,
        
        640950, 645152,
        661953, 666154,        
        };
    check(expect, host_difffilter, expect.size(), 1.f);

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

    float alpha = 1.5f, beta = 1.5f;
    handle.async_convolution_backward_weight(
        covdes, dnnl::algorithm::convolution_direct, alpha, dataTensor, data,
        diffoutTensor, diffout, beta, difffilterTensor, difffilter);
    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_difffilter.data(), difffilter, sizeof(float) * fele_num)
        .wait();

    std::vector<float> expect = {
        189924, 191822,
        199407, 201304,
        
        237330, 239228,
        246813, 248710,
        
        235248, 237914,
        248571, 251236,
        
        301854, 304520,
        315177, 317842,
        
        452172, 455606,
        469335, 472768,
        
        537978, 541412,
        555141, 558574,
        
        535896, 540098,
        556899, 561100,
        
        640902, 645104,
        661905, 666106, 
        };
    check(expect, host_difffilter, expect.size(), 1.f);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
