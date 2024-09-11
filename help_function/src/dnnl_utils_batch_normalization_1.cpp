// ====------ dnnl_utils_batch_normalization_1.cpp --------------===////
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
// test_feature:batch_normalization_mode
// test_feature:batch_normalization_ops
// test_feature:batch_normalization_forward_inference
// test_feature:batch_normalization_forward_inference_ex
// test_feature:batch_normalization_forward_training
// test_feature:batch_normalization_forward_training_ex

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
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, scalebiasTensor;
    handle.create_engine();

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);
    scalebiasTensor.set(dpct::dnnl::memory_format_tag::nchw,
                        dpct::library_data_t::real_float, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_z(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_scale(sbn * sbc * sbh * sbw, 1.0f);
    std::vector<float> host_bias(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rmean(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rvar(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_smean(save * sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_svar(save * sbn * sbc * sbh * sbw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
    }
    for(int i = 0; i < sbn * sbc * sbh * sbw; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    z = (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    scale = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                         q_ct1);
    bias = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                        q_ct1);
    rmean = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                         q_ct1);
    rvar = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                        q_ct1);
    smean = (float *)sycl::malloc_device(
        sizeof(float) * save * sbn * sbc * sbh * sbw, q_ct1);
    svar = (float *)sycl::malloc_device(
        sizeof(float) * save * sbn * sbc * sbh * sbw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1
        .memcpy(scale, host_scale.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1.memcpy(bias, host_bias.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(rmean, host_rmean.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1.memcpy(rvar, host_rvar.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(smean, host_smean.data(),
                sizeof(float) * save * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(svar, host_svar.data(),
                sizeof(float) * save * sbn * sbc * sbh * sbw)
        .wait();

    float alpha = 2.5f, beta = 1.5f, eps = 0.5f;

    auto status = (handle.async_batch_normalization_forward_inference(
                       dpct::dnnl::batch_normalization_mode::per_activation,
                       eps, alpha, dataTensor, data, beta, outTensor, out,
                       scalebiasTensor, scale, bias, smean, svar),
                   0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        1.5, 12.165, 19.1491, 25.0357, 30.3562,
        35.3201, 40.0339, 44.5604, 48.9398, 53.1999,
        57.3607, 61.4372, 65.4411, 69.3815, 73.2658,
        77.1, 80.8893, 84.6378, 88.3491, 92.0265,
        95.6726, 99.2898, 102.88, 106.445, 109.987,
        
        113.507, 117.007, 120.487, 123.949, 127.393,
        130.821, 134.234, 137.632, 141.015, 144.385,
        147.743, 151.088, 154.421, 157.743, 161.053,
        164.354, 167.644, 170.925, 174.196, 177.459,
        180.712, 183.958, 187.195, 190.424, 193.646,
        
        196.86, 200.067, 203.267, 206.46, 209.647,
        212.827, 216.001, 219.169, 222.332, 225.488,
        228.639, 231.784, 234.924, 238.059, 241.189,
        244.314, 247.434, 250.55, 253.661, 256.767,
        259.869, 262.966, 266.06, 269.149, 272.234,
        
        275.315, 278.393, 281.466, 284.536, 287.602,
        290.664, 293.724, 296.779, 299.831, 302.88,
        305.925, 308.968, 312.007, 315.043, 318.076,
        321.106, 324.133, 327.157, 330.178, 333.197,
        336.212, 339.225, 342.236, 345.243, 348.248,

        1.5, 216.289, 335.377, 425.928, 501.761,
        568.322, 628.382, 683.57, 734.934, 783.196,
        828.877, 872.368, 913.969, 953.919, 992.411,
        1029.6, 1065.62, 1100.58, 1134.58, 1167.69,
        1199.99, 1231.53, 1262.38, 1292.58, 1322.17,
        
        1351.19, 1379.68, 1407.66, 1435.17, 1462.23,
        1488.86, 1515.08, 1540.92, 1566.4, 1591.52,
        1616.31, 1640.78, 1664.94, 1688.81, 1712.39,
        1735.7, 1758.75, 1781.55, 1804.11, 1826.43,
        1848.52, 1870.4, 1892.06, 1913.52, 1934.78,
        
        1955.85, 1976.74, 1997.44, 2017.96, 2038.32,
        2058.51, 2078.53, 2098.4, 2118.12, 2137.69,
        2157.11, 2176.39, 2195.54, 2214.55, 2233.42,
        2252.17, 2270.79, 2289.29, 2307.68, 2325.94,
        2344.09, 2362.13, 2380.05, 2397.87, 2415.59,
        
        2433.2, 2450.71, 2468.12, 2485.43, 2502.65,
        2519.78, 2536.81, 2553.75, 2570.61, 2587.38,
        2604.06, 2620.66, 2637.18, 2653.62, 2669.97,
        2686.25, 2702.46, 2718.58, 2734.64, 2750.62,
        2766.52, 2782.36, 2798.13, 2813.83, 2829.46
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
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, scalebiasTensor;
    handle.create_engine();

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 1, sbw = 1;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);
    scalebiasTensor.set(dpct::dnnl::memory_format_tag::nchw,
                        dpct::library_data_t::real_float, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_z(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_scale(sbn * sbc * sbh * sbw, 1.0f);
    std::vector<float> host_bias(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rmean(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rvar(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_smean(save * sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_svar(save * sbn * sbc * sbh * sbw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
    }
    for(int i = 0; i < sbn * sbc * sbh * sbw; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    data =
        (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    z = (float *)sycl::malloc_device(sizeof(float) * in * ic * ih * iw, q_ct1);
    out =
        (float *)sycl::malloc_device(sizeof(float) * on * oc * oh * ow, q_ct1);
    scale = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                         q_ct1);
    bias = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                        q_ct1);
    rmean = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                         q_ct1);
    rvar = (float *)sycl::malloc_device(sizeof(float) * sbn * sbc * sbh * sbw,
                                        q_ct1);
    smean = (float *)sycl::malloc_device(
        sizeof(float) * save * sbn * sbc * sbh * sbw, q_ct1);
    svar = (float *)sycl::malloc_device(
        sizeof(float) * save * sbn * sbc * sbh * sbw, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw)
        .wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * in * ic * ih * iw).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow)
        .wait();
    q_ct1
        .memcpy(scale, host_scale.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1.memcpy(bias, host_bias.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(rmean, host_rmean.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1.memcpy(rvar, host_rvar.data(), sizeof(float) * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(smean, host_smean.data(),
                sizeof(float) * save * sbn * sbc * sbh * sbw)
        .wait();
    q_ct1
        .memcpy(svar, host_svar.data(),
                sizeof(float) * save * sbn * sbc * sbh * sbw)
        .wait();

    float alpha = 2.5f, beta = 1.5f, eps = 0.5f;

    auto status = (handle.async_batch_normalization_forward_inference(
                       dpct::dnnl::batch_normalization_mode::spatial, eps,
                       alpha, dataTensor, data, beta, outTensor, out,
                       scalebiasTensor, scale, bias, smean, svar),
                   0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow).wait();
    std::vector<float> expect = {
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        61.1548, 63.196, 65.2372, 67.2785, 69.3197,
        71.361, 73.4022, 75.4435, 77.4847, 79.5259,
        81.5672, 83.6084, 85.6497, 87.6909, 89.7321,
        91.7734, 93.8146, 95.8559, 97.8971, 99.9383,
        101.98, 104.021, 106.062, 108.103, 110.145,
        
        170.938, 174.101, 177.263, 180.425, 183.588,
        186.75, 189.912, 193.074, 196.237, 199.399,
        202.561, 205.723, 208.886, 212.048, 215.21,
        218.373, 221.535, 224.697, 227.859, 231.022,
        234.184, 237.346, 240.509, 243.671, 246.833,
        
        313.678, 317.687, 321.696, 325.705, 329.714,
        333.722, 337.731, 341.74, 345.749, 349.758,
        353.767, 357.776, 361.785, 365.794, 369.803,
        373.812, 377.82, 381.829, 385.838, 389.847,
        393.856, 397.865, 401.874, 405.883, 409.892,

        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        265.279, 267.32, 269.361, 271.403, 273.444,
        275.485, 277.526, 279.568, 281.609, 283.65,
        285.691, 287.733, 289.774, 291.815, 293.856,
        295.898, 297.939, 299.98, 302.021, 304.062,
        306.104, 308.145, 310.186, 312.227, 314.269,
        
        487.166, 490.328, 493.491, 496.653, 499.815,
        502.978, 506.14, 509.302, 512.464, 515.627,
        518.789, 521.951, 525.114, 528.276, 531.438,
        534.6, 537.763, 540.925, 544.087, 547.249,
        550.412, 553.574, 556.736, 559.899, 563.061,
        
        714.57, 718.579, 722.588, 726.596, 730.605,
        734.614, 738.623, 742.632, 746.641, 750.65,
        754.659, 758.668, 762.677, 766.686, 770.694,
        774.703, 778.712, 782.721, 786.73, 790.739,
        794.748, 798.757, 802.766, 806.775, 810.784        
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
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, scalebiasTensor,
        additionTensor;
    handle.create_engine();

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    dataTensor.set(dpct::dnnl::memory_format_tag::nchw,
                   dpct::library_data_t::real_float, in, ic, ih, iw);
    outTensor.set(dpct::dnnl::memory_format_tag::nchw,
                  dpct::library_data_t::real_float, on, oc, oh, ow);
    scalebiasTensor.set(dpct::dnnl::memory_format_tag::nchw,
                        dpct::library_data_t::real_float, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(ele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_rmean[i] = i;
        host_rvar[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    data = sycl::malloc_device<float>(ele_num, q_ct1);
    z = sycl::malloc_device<float>(ele_num, q_ct1);
    out = sycl::malloc_device<float>(oele_num, q_ct1);
    scale = sycl::malloc_device<float>(sele_num, q_ct1);
    bias = sycl::malloc_device<float>(sele_num, q_ct1);
    rmean = sycl::malloc_device<float>(sele_num, q_ct1);
    rvar = sycl::malloc_device<float>(sele_num, q_ct1);
    smean =
        (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);
    svar = (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * ele_num).wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * ele_num).wait();
    q_ct1.memcpy(out, host_out.data(), sizeof(float) * oele_num).wait();
    q_ct1.memcpy(scale, host_scale.data(), sizeof(float) * sele_num).wait();
    q_ct1.memcpy(bias, host_bias.data(), sizeof(float) * sele_num).wait();
    q_ct1.memcpy(rmean, host_rmean.data(), sizeof(float) * sele_num).wait();
    q_ct1.memcpy(rvar, host_rvar.data(), sizeof(float) * sele_num).wait();
    q_ct1.memcpy(smean, host_smean.data(), sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(svar, host_svar.data(), sizeof(float) * save * sele_num)
        .wait();

    float alpha = 2.5f, beta = 1.5f, eps = 1.f;
    double factor = 0.1f;

    auto status =
        (handle.async_batch_normalization_forward_training(
             dpct::dnnl::batch_normalization_mode::per_activation, eps, factor,
             alpha, dataTensor, data, beta, outTensor, out, scalebiasTensor,
             scale, bias, rmean, rvar, smean, svar),
         0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * oele_num).wait();
    q_ct1.memcpy(host_rmean.data(), rmean, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_rvar.data(), rvar, sizeof(float) * save * sele_num)
        .wait();
    std::vector<float> expect = {
        1.5, 1.5005, 1.501, 1.5015, 1.502,
        1.5025, 1.503, 1.5035, 1.504, 1.5045,
        1.505, 1.5055, 1.506, 1.5065, 1.507,
        1.5075, 1.508, 1.5085, 1.509, 1.5095,
        1.51, 1.5105, 1.511, 1.5115, 1.512,
        
        1.5125, 1.513, 1.5135, 1.514, 1.5145,
        1.515, 1.5155, 1.516, 1.5165, 1.517,
        1.5175, 1.518, 1.5185, 1.519, 1.5195,
        1.52, 1.5205, 1.521, 1.5215, 1.522,
        1.5225, 1.523, 1.5235, 1.524, 1.5245,
        
        1.525, 1.5255, 1.526, 1.5265, 1.527,
        1.5275, 1.528, 1.5285, 1.529, 1.5295,
        1.53, 1.5305, 1.531, 1.5315, 1.532,
        1.5325, 1.533, 1.5335, 1.534, 1.5345,
        1.535, 1.5355, 1.536, 1.5365, 1.537,
        
        1.5375, 1.538, 1.5385, 1.539, 1.5395,
        1.54, 1.5405, 1.541, 1.5415, 1.542,
        1.5425, 1.543, 1.5435, 1.544, 1.54449,
        1.54499, 1.54549, 1.54599, 1.54649, 1.54699,
        1.54749, 1.54799, 1.54849, 1.54899, 1.54949,

        1.5, 6.4995, 11.499, 16.4985, 21.498,
        26.4975, 31.497, 36.4965, 41.496, 46.4955,
        51.495, 56.4945, 61.494, 66.4935, 71.493,
        76.4925, 81.492, 86.4915, 91.491, 96.4905,
        101.49, 106.49, 111.489, 116.488, 121.488,
        
        126.487, 131.487, 136.487, 141.486, 146.486,
        151.485, 156.484, 161.484, 166.484, 171.483,
        176.483, 181.482, 186.482, 191.481, 196.48,
        201.48, 206.479, 211.479, 216.479, 221.478,
        226.477, 231.477, 236.477, 241.476, 246.476,
        
        251.475, 256.474, 261.474, 266.474, 271.473,
        276.473, 281.472, 286.471, 291.471, 296.471,
        301.47, 306.47, 311.469, 316.469, 321.468,
        326.467, 331.467, 336.466, 341.466, 346.466,
        351.465, 356.464, 361.464, 366.464, 371.463,
        
        376.462, 381.462, 386.462, 391.461, 396.461,
        401.46, 406.459, 411.459, 416.458, 421.458,
        426.458, 431.457, 436.457, 441.456, 446.456,
        451.455, 456.454, 461.454, 466.453, 471.453,
        476.453, 481.452, 486.452, 491.451, 496.451
        };
    std::vector<float> expect_rmean = {
        5.4, 6.4, 7.4, 8.4, 9.4,
        10.4, 11.4, 12.4, 13.4, 14.4,
        15.4, 16.4, 17.4, 18.4, 19.4,
        20.4, 21.4, 22.4, 23.4, 24.4,
        25.4, 26.4, 27.4, 28.4, 29.4,
        
        30.4, 31.4, 32.4, 33.4, 34.4,
        35.4, 36.4, 37.4, 38.4, 39.4,
        40.4, 41.4, 42.4, 43.4, 44.4,
        45.4, 46.4, 47.4, 48.4, 49.4,
        50.4, 51.4, 52.4, 53.4, 54.4,
        
        55.4, 56.4, 57.4, 58.4, 59.4,
        60.4, 61.4, 62.4, 63.4, 64.4,
        65.4, 66.4, 67.4, 68.4, 69.4,
        70.4, 71.4, 72.4, 73.4, 74.4,
        75.4, 76.4, 77.4, 78.4, 79.4,
        
        80.4, 81.4, 82.4, 83.4, 84.4,
        85.4, 86.4, 87.4, 88.4, 89.4,
        90.4, 91.4, 92.4, 93.4, 94.4,
        95.4, 96.4, 97.4, 98.4, 99.4,
        100.4, 101.4, 102.4, 103.4, 104.4        
        };

    std::vector<float> expect_rvar = {
        500, 500.9, 501.8, 502.7, 503.6,
        504.5, 505.4, 506.3, 507.2, 508.1,
        509, 509.9, 510.8, 511.7, 512.6,
        513.5, 514.4, 515.3, 516.2, 517.1,
        518, 518.9, 519.8, 520.7, 521.6,
        
        522.5, 523.4, 524.3, 525.2, 526.1,
        527, 527.9, 528.8, 529.7, 530.6,
        531.5, 532.4, 533.3, 534.2, 535.1,
        536, 536.9, 537.8, 538.7, 539.6,
        540.5, 541.4, 542.3, 543.2, 544.1,
        
        545, 545.9, 546.8, 547.7, 548.6,
        549.5, 550.4, 551.3, 552.2, 553.1,
        554, 554.9, 555.8, 556.7, 557.6,
        558.5, 559.4, 560.3, 561.2, 562.1,
        563, 563.9, 564.8, 565.7, 566.6,
        
        567.5, 568.4, 569.3, 570.2, 571.1,
        572, 572.9, 573.8, 574.7, 575.6,
        576.5, 577.4, 578.3, 579.2, 580.1,
        581, 581.9, 582.8, 583.7, 584.6,
        585.5, 586.4, 587.3, 588.2, 589.1        
        };
    check(expect, host_out, expect.size(), 1e-1);
    check(expect_rmean, host_rmean, expect_rmean.size(), 1e-1);
    check(expect_rvar, host_rvar, expect_rvar.size(), 1e-1);

    sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}



int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    test3<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
