// ====------ dnnl_utils_normalization_1.cpp --------------===////
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
    std::vector<float> host_z(oele_num, 1.0f);
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
        host_z[i] = 10;
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
    z = sycl::malloc_device<float>(oele_num, q_ct1);
    out = sycl::malloc_device<float>(oele_num, q_ct1);
    scale = sycl::malloc_device<float>(sele_num, q_ct1);
    bias = sycl::malloc_device<float>(sele_num, q_ct1);
    rmean = sycl::malloc_device<float>(sele_num, q_ct1);
    rvar = sycl::malloc_device<float>(sele_num, q_ct1);
    smean =
        (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);
    svar = (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * ele_num).wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * oele_num).wait();
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
    double factor = 0.5f;
    dpct::dnnl::activation_desc ActivationDesc;
                ActivationDesc.set(dnnl::algorithm::eltwise_relu_use_dst_for_bwd, 0.0f);

        auto status =
        (handle.async_batch_normalization_forward_inference(
             dpct::dnnl::batch_normalization_mode::per_activation,
             dpct::dnnl::batch_normalization_ops::none, ActivationDesc, eps,
             alpha, dataTensor, data, beta, outTensor, out, dataTensor, z,
             scalebiasTensor, scale, bias, scalebiasTensor, smean, svar),
         0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * oele_num).wait();

    std::vector<float> expect = {
        1.5, 11.0711, 18.047, 24, 29.3885,
        34.4124, 39.1779, 43.7487, 48.1667, 52.4605,
        56.6511, 60.7543, 64.782, 68.744, 72.6478,
        76.5, 80.3057, 84.0694, 87.7948, 91.4853,
        95.1436, 98.7721, 102.373, 105.949, 109.5,
        
        113.029, 116.537, 120.025, 123.495, 126.947,
        130.382, 133.801, 137.205, 140.595, 143.97,
        147.333, 150.684, 154.022, 157.349, 160.664,
        163.969, 167.264, 170.549, 173.825, 177.091,
        180.349, 183.598, 186.839, 190.071, 193.296,
        
        196.514, 199.724, 202.927, 206.124, 209.314,
        212.497, 215.674, 218.845, 222.01, 225.169,
        228.322, 231.47, 234.613, 237.75, 240.882,
        244.009, 247.132, 250.249, 253.362, 256.471,
        259.575, 262.674, 265.77, 268.861, 271.948,
        
        275.031, 278.11, 281.185, 284.257, 287.325,
        290.389, 293.45, 296.507, 299.56, 302.611,
        305.658, 308.702, 311.742, 314.78, 317.814,
        320.846, 323.874, 326.9, 329.922, 332.942,
        335.959, 338.973, 341.985, 344.994, 348,

        1.5, 187.848, 306.722, 399, 476.602,
        544.723, 606.125, 662.467, 714.833, 763.973,
        810.43, 854.611, 896.832, 937.343, 976.344,
        1014, 1050.45, 1085.8, 1120.17, 1153.62,
        1186.23, 1218.08, 1249.2, 1279.66, 1309.5,
        
        1338.75, 1367.46, 1395.66, 1423.36, 1450.61,
        1477.42, 1503.82, 1529.83, 1555.46, 1580.73,
        1605.67, 1630.27, 1654.57, 1678.57, 1702.27,
        1725.71, 1748.87, 1771.78, 1794.45, 1816.87,
        1839.07, 1861.05, 1882.81, 1904.36, 1925.71,
        
        1946.86, 1967.83, 1988.61, 2009.22, 2029.65,
        2049.92, 2070.02, 2089.96, 2109.75, 2129.39,
        2148.88, 2168.22, 2187.43, 2206.5, 2225.44,
        2244.25, 2262.93, 2281.49, 2299.92, 2318.24,
        2336.44, 2354.53, 2372.51, 2390.38, 2408.14,
        
        2425.8, 2443.36, 2460.82, 2478.18, 2495.44,
        2512.61, 2529.69, 2546.67, 2563.57, 2580.38,
        2597.1, 2613.74, 2630.3, 2646.78, 2663.17,
        2679.49, 2695.73, 2711.89, 2727.98, 2743.99,
        2759.93, 2775.8, 2791.6, 2807.34, 2823,        
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
    std::vector<float> host_z(oele_num, 1.0f);
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
        host_z[i] = 10;
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
    z = sycl::malloc_device<float>(oele_num, q_ct1);
    out = sycl::malloc_device<float>(oele_num, q_ct1);
    scale = sycl::malloc_device<float>(sele_num, q_ct1);
    bias = sycl::malloc_device<float>(sele_num, q_ct1);
    rmean = sycl::malloc_device<float>(sele_num, q_ct1);
    rvar = sycl::malloc_device<float>(sele_num, q_ct1);
    smean =
        (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);
    svar = (float *)sycl::malloc_device(sizeof(float) * save * sele_num, q_ct1);

    q_ct1.memcpy(data, host_data.data(), sizeof(float) * ele_num).wait();
    q_ct1.memcpy(z, host_z.data(), sizeof(float) * oele_num).wait();
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
    double factor = 0.5f;
    dpct::dnnl::activation_desc ActivationDesc;
                ActivationDesc.set(dnnl::algorithm::eltwise_relu_use_dst_for_bwd, 0.0f);

        auto status =
        (handle.async_batch_normalization_forward_inference(
             dpct::dnnl::batch_normalization_mode::spatial,
             dpct::dnnl::batch_normalization_ops::none, ActivationDesc, eps,
             alpha, dataTensor, data, beta, outTensor, out, dataTensor, z,
             scalebiasTensor, scale, bias, scalebiasTensor, smean, svar),
         0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * oele_num).wait();

    std::vector<float> expect = {
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        53.4975, 55.2652, 57.033, 58.8008, 60.5685,
        62.3363, 64.1041, 65.8718, 67.6396, 69.4074,
        71.1751, 72.9429, 74.7107, 76.4784, 78.2462,
        80.014, 81.7817, 83.5495, 85.3173, 87.085,
        88.8528, 90.6206, 92.3883, 94.1561, 95.9239,
        
        156.611, 159.498, 162.385, 165.271, 168.158,
        171.045, 173.932, 176.818, 179.705, 182.592,
        185.479, 188.365, 191.252, 194.139, 197.026,
        199.912, 202.799, 205.686, 208.573, 211.459,
        214.346, 217.233, 220.12, 223.006, 225.893,
        
        294, 297.75, 301.5, 305.25, 309,
        312.75, 316.5, 320.25, 324, 327.75,
        331.5, 335.25, 339, 342.75, 346.5,
        350.25, 354, 357.75, 361.5, 365.25,
        369, 372.75, 376.5, 380.25, 384,

        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        230.274, 232.042, 233.81, 235.577, 237.345,
        239.113, 240.881, 242.649, 244.416, 246.184,
        247.952, 249.72, 251.487, 253.255, 255.023,
        256.791, 258.558, 260.326, 262.094, 263.862,
        265.629, 267.397, 269.165, 270.933, 272.701,
        
        445.286, 448.173, 451.06, 453.946, 456.833,
        459.72, 462.607, 465.493, 468.38, 471.267,
        474.154, 477.04, 479.927, 482.814, 485.701,
        488.587, 491.474, 494.361, 497.248, 500.134,
        503.021, 505.908, 508.795, 511.681, 514.568,

        669, 672.75, 676.5, 680.25, 684,
        687.75, 691.5, 695.25, 699, 702.75,
        706.5, 710.25, 714, 717.75, 721.5,
        725.25, 729, 732.75, 736.5, 740.25,
        744, 747.75, 751.5, 755.25, 759,              
        };
    check(expect, host_out, expect.size(), 1e-1);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
