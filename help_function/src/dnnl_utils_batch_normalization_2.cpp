// ====------ dnnl_utils_batch_normalization_2.cpp --------------===////
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
void test4() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    dpct::dnnl::engine_ext handle;
    dpct::dnnl::memory_desc_ext dataTensor, outTensor, scalebiasTensor,
        additionTensor;
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
        auto status = (handle.async_batch_normalization_forward_training(
                       dpct::dnnl::batch_normalization_mode::spatial, eps,
                       factor, alpha, dataTensor, data, beta, outTensor, out,
                       scalebiasTensor, scale, bias, rmean, rvar, smean, svar),
                   0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * oele_num).wait();
    q_ct1.memcpy(host_rmean.data(), rmean, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_rvar.data(), rvar, sizeof(float) * save * sele_num)
        .wait();
    std::vector<float> expect = {
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        0.932347, 0.981825, 1.0313, 1.08078, 1.13026,
        1.17974, 1.22922, 1.27869, 1.32817, 1.37765,
        1.42713, 1.47661, 1.52609, 1.57556, 1.62504,
        1.67452, 1.724, 1.77348, 1.82296, 1.87243,
        1.92191, 1.97139, 2.02087, 2.07035, 2.11983,
        
        0.364693, 0.46365, 0.562607, 0.661563, 0.76052,
        0.859476, 0.958433, 1.05739, 1.15635, 1.2553,
        1.35426, 1.45322, 1.55217, 1.65113, 1.75009,
        1.84904, 1.948, 2.04695, 2.14591, 2.24487,
        2.34382, 2.44278, 2.54174, 2.64069, 2.73965,
        
        -0.202961, -0.0545259, 0.093909, 0.242344, 0.390779,
        0.539214, 0.687648, 0.836083, 0.984518, 1.13295,
        1.28139, 1.42982, 1.57826, 1.72669, 1.87513,
        2.02356, 2.172, 2.32043, 2.46887, 2.6173,
        2.76574, 2.91417, 3.06261, 3.21104, 3.35948,

        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        5.88017, 5.92965, 5.97913, 6.02861, 6.07809,
        6.12757, 6.17704, 6.22652, 6.276, 6.32548,
        6.37496, 6.42444, 6.47391, 6.52339, 6.57287,
        6.62235, 6.67183, 6.72131, 6.77078, 6.82026,
        6.86974, 6.91922, 6.9687, 7.01818, 7.06765,
        
        10.2603, 10.3593, 10.4583, 10.5572, 10.6562,
        10.7551, 10.8541, 10.953, 11.052, 11.151,
        11.2499, 11.3489, 11.4478, 11.5468, 11.6457,
        11.7447, 11.8437, 11.9426, 12.0416, 12.1405,
        12.2395, 12.3384, 12.4374, 12.5364, 12.6353,
        
        14.6405, 14.789, 14.9374, 15.0858, 15.2343,
        15.3827, 15.5311, 15.6796, 15.828, 15.9764,
        16.1249, 16.2733, 16.4217, 16.5702, 16.7186,
        16.867, 17.0155, 17.1639, 17.3124, 17.4608,
        17.6092, 17.7577, 17.9061, 18.0545, 18.203,
        };
    std::vector<float> expect_rmean = {
        6.6,

        10,
        
        13.4,
        
        16.8,
        };

    std::vector<float> expect_rvar = {
        260.408,

        261.308,
        
        262.208,
        
        263.108,              
        };
    check(expect, host_out, expect.size(), 1e-1);
    check(expect_rmean, host_rmean, expect_rmean.size(), 1e-1);
    check(expect_rvar, host_rvar, expect_rvar.size(), 1e-1);
        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test5() {
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

    float *workspace, *reservespace;
    size_t workspace_size, reservespace_size;

    workspace_size = 0;

    reservespace_size = handle.get_batch_normalization_workspace_size(
        dpct::dnnl::batch_normalization_ops::activation, dataTensor);

    workspace = (float *)sycl::malloc_device(workspace_size, q_ct1);
    reservespace = (float *)sycl::malloc_device(reservespace_size, q_ct1);

        auto status =
        (handle.async_batch_normalization_forward_training(
             dpct::dnnl::batch_normalization_mode::per_activation,
             dpct::dnnl::batch_normalization_ops::activation, ActivationDesc,
             eps, factor, alpha, dataTensor, data, beta, outTensor, out,
             outTensor, z, scalebiasTensor, scale, bias, rmean, rvar, smean,
             svar, reservespace_size, reservespace),
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
        476.453, 481.452, 486.452, 491.451, 496.451,
        };
    std::vector<float> expect_rmean = {
        27, 28, 29, 30, 31,
        32, 33, 34, 35, 36,
        37, 38, 39, 40, 41,
        42, 43, 44, 45, 46,
        47, 48, 49, 50, 51,
        
        52, 53, 54, 55, 56,
        57, 58, 59, 60, 61,
        62, 63, 64, 65, 66,
        67, 68, 69, 70, 71,
        72, 73, 74, 75, 76,
        
        77, 78, 79, 80, 81,
        82, 83, 84, 85, 86,
        87, 88, 89, 90, 91,
        92, 93, 94, 95, 96,
        97, 98, 99, 100, 101,
        
        102, 103, 104, 105, 106,
        107, 108, 109, 110, 111,
        112, 113, 114, 115, 116,
        117, 118, 119, 120, 121,
        122, 123, 124, 125, 126,        
        };

    std::vector<float> expect_rvar = {
        2500, 2500.5, 2501, 2501.5, 2502,
        2502.5, 2503, 2503.5, 2504, 2504.5,
        2505, 2505.5, 2506, 2506.5, 2507,
        2507.5, 2508, 2508.5, 2509, 2509.5,
        2510, 2510.5, 2511, 2511.5, 2512,
        
        2512.5, 2513, 2513.5, 2514, 2514.5,
        2515, 2515.5, 2516, 2516.5, 2517,
        2517.5, 2518, 2518.5, 2519, 2519.5,
        2520, 2520.5, 2521, 2521.5, 2522,
        2522.5, 2523, 2523.5, 2524, 2524.5,
        
        2525, 2525.5, 2526, 2526.5, 2527,
        2527.5, 2528, 2528.5, 2529, 2529.5,
        2530, 2530.5, 2531, 2531.5, 2532,
        2532.5, 2533, 2533.5, 2534, 2534.5,
        2535, 2535.5, 2536, 2536.5, 2537,
        
        2537.5, 2538, 2538.5, 2539, 2539.5,
        2540, 2540.5, 2541, 2541.5, 2542,
        2542.5, 2543, 2543.5, 2544, 2544.5,
        2545, 2545.5, 2546, 2546.5, 2547,
        2547.5, 2548, 2548.5, 2549, 2549.5,                  
        };
    check(expect, host_out, expect.size(), 1e-1);
    check(expect_rmean, host_rmean, expect_rmean.size(), 1e-1);
    check(expect_rvar, host_rvar, expect_rvar.size(), 1e-1);

        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test6() {
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
    float *diffout, *diffdata, *diffscale, *diffbias;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(ele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);
    std::vector<float> host_diffout(oele_num, 0.f);
    std::vector<float> host_diffdata(ele_num, 0.f);
    std::vector<float> host_diffscale(sele_num, 1.0f);
    std::vector<float> host_diffbias(sele_num, 0.0f);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  1.5f * i + 4.f;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = 1.f;
        host_diffout[i] = 100 * i;
        host_diffdata[i] = 0.f;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_rmean[i] = i;
        host_rvar[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
        host_diffscale[i] = 0.f;
        host_diffbias[i] = 0.f;
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
    diffout = sycl::malloc_device<float>(oele_num, q_ct1);
    diffdata = sycl::malloc_device<float>(ele_num, q_ct1);
    diffscale = sycl::malloc_device<float>(sele_num, q_ct1);
    diffbias = sycl::malloc_device<float>(sele_num, q_ct1);

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
    q_ct1.memcpy(diffout, host_diffout.data(), sizeof(float) * oele_num).wait();
    q_ct1.memcpy(diffdata, host_diffdata.data(), sizeof(float) * oele_num)
        .wait();
    q_ct1.memcpy(diffscale, host_diffscale.data(), sizeof(float) * sele_num)
        .wait();
    q_ct1.memcpy(diffbias, host_diffbias.data(), sizeof(float) * sele_num)
        .wait();
    float alpha = 2.5f, beta = 1.5f, eps = 1.f;
    double factor = 0.1f;
        auto status =
        (handle.async_batch_normalization_forward_training(
             dpct::dnnl::batch_normalization_mode::per_activation, eps, factor,
             alpha, dataTensor, data, beta, outTensor, out, scalebiasTensor,
             scale, bias, rmean, rvar, smean, svar),
         0);
        status = (handle.async_batch_normalization_backward(
                  dpct::dnnl::batch_normalization_mode::per_activation, eps,
                  alpha, dataTensor, data, outTensor, diffout, beta, dataTensor,
                  diffdata, alpha, scalebiasTensor, scale, beta, diffscale,
                  diffbias, smean, svar),
              0);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num)
        .wait();
    q_ct1
        .memcpy(host_diffscale.data(), diffscale,
                sizeof(float) * save * sele_num)
        .wait();
    q_ct1
        .memcpy(host_diffbias.data(), diffbias, sizeof(float) * save * sele_num)
        .wait();

    std::vector<float> expect = {
        0, -0.0296198, -0.0592395, -0.0888593, -0.118479,
        -0.148099, -0.177719, -0.207338, -0.236958, -0.266578,
        -0.296198, -0.325817, -0.355437, -0.385057, -0.414677,
        -0.444296, -0.473916, -0.503536, -0.533156, -0.562775,
        -0.592395, -0.622015, -0.651635, -0.681255, -0.710874,
        
        -0.740494, -0.770114, -0.799734, -0.829353, -0.858973,
        -0.888593, -0.918213, -0.947832, -0.977452, -1.00707,
        -1.03669, -1.06631, -1.09593, -1.12555, -1.15517,
        -1.18479, -1.21441, -1.24403, -1.27365, -1.30327,
        -1.33289, -1.36251, -1.39213, -1.42175, -1.45137,
        
        -1.48099, -1.51061, -1.54023, -1.56985, -1.59947,
        -1.62909, -1.65871, -1.68833, -1.71795, -1.74757,
        -1.77719, -1.80681, -1.83643, -1.86604, -1.89566,
        -1.92528, -1.9549, -1.98452, -2.01414, -2.04376,
        -2.07338, -2.103, -2.13262, -2.16224, -2.19186,
        
        -2.22148, -2.2511, -2.28072, -2.31034, -2.33996,
        -2.36958, -2.3992, -2.42882, -2.45844, -2.48806,
        -2.51768, -2.5473, -2.57692, -2.60654, -2.63616,
        -2.66578, -2.6954, -2.72502, -2.75464, -2.78426,
        -2.81388, -2.8435, -2.87312, -2.90274, -2.93236,

        0, 0.0296198, 0.0592395, 0.0888593, 0.118479,
        0.148099, 0.177719, 0.207338, 0.236958, 0.266578,
        0.296198, 0.325817, 0.355437, 0.385057, 0.414677,
        0.444296, 0.473916, 0.503536, 0.533156, 0.562775,
        0.592395, 0.622015, 0.651635, 0.681255, 0.710874,
        
        0.740494, 0.770114, 0.799734, 0.829353, 0.858973,
        0.888593, 0.918213, 0.947832, 0.977452, 1.00707,
        1.03669, 1.06631, 1.09593, 1.12555, 1.15517,
        1.18479, 1.21441, 1.24403, 1.27365, 1.30327,
        1.33289, 1.36251, 1.39213, 1.42175, 1.45137,
        
        1.48099, 1.51061, 1.54023, 1.56985, 1.59947,
        1.62909, 1.65871, 1.68833, 1.71795, 1.74757,
        1.77719, 1.80681, 1.83643, 1.86604, 1.89566,
        1.92528, 1.9549, 1.98452, 2.01414, 2.04376,
        2.07338, 2.103, 2.13262, 2.16224, 2.19186,
        
        2.22148, 2.2511, 2.28072, 2.31034, 2.33996,
        2.36958, 2.3992, 2.42882, 2.45844, 2.48806,
        2.51768, 2.5473, 2.57692, 2.60654, 2.63616,
        2.66578, 2.6954, 2.72502, 2.75464, 2.78426,
        2.81388, 2.8435, 2.87312, 2.90274, 2.93236,
        };

    std::vector<float> expect_diffscale = {
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,
        24997.8, 24997.8, 24997.8, 24997.8, 24997.8,        
        };

    std::vector<float> expect_diffbias = {
        25000, 25500, 26000, 26500, 27000,
        27500, 28000, 28500, 29000, 29500,
        30000, 30500, 31000, 31500, 32000,
        32500, 33000, 33500, 34000, 34500,
        35000, 35500, 36000, 36500, 37000,
        
        37500, 38000, 38500, 39000, 39500,
        40000, 40500, 41000, 41500, 42000,
        42500, 43000, 43500, 44000, 44500,
        45000, 45500, 46000, 46500, 47000,
        47500, 48000, 48500, 49000, 49500,
        
        50000, 50500, 51000, 51500, 52000,
        52500, 53000, 53500, 54000, 54500,
        55000, 55500, 56000, 56500, 57000,
        57500, 58000, 58500, 59000, 59500,
        60000, 60500, 61000, 61500, 62000,
        
        62500, 63000, 63500, 64000, 64500,
        65000, 65500, 66000, 66500, 67000,
        67500, 68000, 68500, 69000, 69500,
        70000, 70500, 71000, 71500, 72000,
        72500, 73000, 73500, 74000, 74500,
        };
    check(expect, host_diffdata, expect.size(), 1e-1);
    check(expect_diffscale, host_diffscale, expect_diffscale.size(), 1e-1);
    check(expect_diffbias, host_diffbias, expect_diffbias.size(), 1e-1);
        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}



int main() {
    test4<dpct::library_data_t::real_float>();
    test5<dpct::library_data_t::real_float>();
    test6<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}
