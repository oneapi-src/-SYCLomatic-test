// ====------ dnnl_utils_batch_normalization_3.cpp --------------===////
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
// test_feature:batch_normalization_backward
// test_feature:batch_normalization_backward_ex

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
void test7() {
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
        auto status = (handle.async_batch_normalization_forward_training(
                       dpct::dnnl::batch_normalization_mode::spatial, eps,
                       factor, alpha, dataTensor, data, beta, outTensor, out,
                       scalebiasTensor, scale, bias, rmean, rvar, smean, svar),
                   0);
        status = (handle.async_batch_normalization_backward(
                  dpct::dnnl::batch_normalization_mode::spatial, eps, alpha,
                  dataTensor, data, outTensor, diffout, beta, dataTensor,
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
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        
        -0.035614, -0.0350037, -0.0344543, -0.0338745, -0.0332947,
        -0.0327148, -0.032135, -0.0315857, -0.0310059, -0.030426,
        -0.0298462, -0.0292664, -0.0286865, -0.0281372, -0.0275574,
        -0.0269775, -0.0263977, -0.0258179, -0.0252686, -0.0246887,
        -0.0241089, -0.0235291, -0.0229492, -0.0223999, -0.0218201,
        
        -0.071228, -0.0700073, -0.0689087, -0.067749, -0.0665894,
        -0.0654297, -0.06427, -0.0631714, -0.0620117, -0.0608521,
        -0.0596924, -0.0585327, -0.057373, -0.0562744, -0.0551147,
        -0.0539551, -0.0527954, -0.0516357, -0.0505371, -0.0493774,
        -0.0482178, -0.0470581, -0.0458984, -0.0447998, -0.0436401,
        
        -0.106812, -0.105103, -0.103394, -0.101685, -0.0999756,
        -0.0982666, -0.0965576, -0.0948486, -0.0931396, -0.0914307,
        -0.0895996, -0.0878906, -0.0861816, -0.0844727, -0.0827637,
        -0.0810547, -0.0793457, -0.0775146, -0.0758057, -0.0740967,
        -0.0723877, -0.0706787, -0.0689697, -0.0672607, -0.0654297,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        
        0.0218506, 0.0224304, 0.0229797, 0.0235596, 0.0241394,
        0.0247192, 0.0252991, 0.0258484, 0.0264282, 0.0270081,
        0.0275879, 0.0281677, 0.028717, 0.0292969, 0.0298767,
        0.0304565, 0.0310364, 0.0315857, 0.0321655, 0.0327454,
        0.0333252, 0.033905, 0.0344849, 0.0350342, 0.0356445,
        
        0.0436401, 0.0447998, 0.0459595, 0.0471191, 0.0482788,
        0.0494385, 0.0505981, 0.0516968, 0.0528564, 0.0540161,
        0.0551758, 0.0563354, 0.0574341, 0.0585938, 0.0597534,
        0.0609131, 0.0620728, 0.0631714, 0.0643311, 0.0654907,
        0.0666504, 0.0678101, 0.0689697, 0.0700684, 0.071228,
        
        0.0654297, 0.0672607, 0.0689697, 0.0706787, 0.0723877,
        0.0740967, 0.0758057, 0.0775146, 0.0792236, 0.0809326,
        0.0827637, 0.0844727, 0.0861816, 0.0878906, 0.0895996,
        0.0913086, 0.0930176, 0.0947266, 0.0964355, 0.0981445,
        0.0999756, 0.101685, 0.103394, 0.105103, 0.106812,
        };

    std::vector<float> expect_diffscale = {
        631412,

        631412,
        
        631412,
        
        631412,              
        };

    std::vector<float> expect_diffbias = {
        775000,

        1.0875E6,
        
        1.4E6,
        
        1.7125E6,
        };
    check(expect, host_diffdata, expect.size(), 1e-1);
    check(expect_diffscale, host_diffscale, expect_diffscale.size(), 1.f);
    check(expect_diffbias, host_diffbias, expect_diffbias.size(), 1.f);
        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test8() {
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
    float *diffout, *diffdata, *diffscale, *diffbias, *diffz;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(oele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);
    std::vector<float> host_diffout(oele_num, 0.f);
    std::vector<float> host_diffz(oele_num, 0.f);
    std::vector<float> host_diffdata(ele_num, 0.f);
    std::vector<float> host_diffscale(sele_num, 1.0f);
    std::vector<float> host_diffbias(sele_num, 0.0f);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = i;
        host_diffout[i] = 100 * i + 1.f;
        host_diffdata[i] = 0.f;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i + 4.f;
        host_bias[i] = i + 4.f;
        host_rmean[i] = i + 4.f;
        host_rvar[i] = i + 4.f;
        host_smean[i] = i + 4.f;
        host_svar[i] = i + 4.f;
        host_diffscale[i] = i + 4.f;
        host_diffbias[i] = i + 4.f;
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
    diffout = sycl::malloc_device<float>(oele_num, q_ct1);
    diffz = sycl::malloc_device<float>(oele_num, q_ct1);
    diffdata = sycl::malloc_device<float>(ele_num, q_ct1);
    diffscale = sycl::malloc_device<float>(sele_num, q_ct1);
    diffbias = sycl::malloc_device<float>(sele_num, q_ct1);

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
    q_ct1.memcpy(diffdata, host_diffdata.data(), sizeof(float) * oele_num)
        .wait();
    q_ct1.memcpy(diffout, host_diffout.data(), sizeof(float) * oele_num).wait();
    q_ct1.memcpy(diffscale, host_diffscale.data(), sizeof(float) * sele_num)
        .wait();
    q_ct1.memcpy(diffbias, host_diffbias.data(), sizeof(float) * sele_num)
        .wait();
    float alpha = 2.5f, beta = 1.5f, eps = 1.f;
    double factor = 0.5f;

    dpct::dnnl::activation_desc ActivationDesc;
                ActivationDesc.set(dnnl::algorithm::eltwise_relu_use_dst_for_bwd, 0.0f);

    float *workspace, *reservespace, *breservespace;
    size_t workspace_size, reservespace_size;

    workspace_size = 0;

    reservespace_size = handle.get_batch_normalization_workspace_size(
        dpct::dnnl::batch_normalization_ops::activation, dataTensor);

    workspace = (float *)sycl::malloc_device(workspace_size, q_ct1);
    reservespace = (float *)sycl::malloc_device(reservespace_size, q_ct1);
    breservespace = (float *)sycl::malloc_device(reservespace_size, q_ct1);

        auto status =
        (handle.async_batch_normalization_forward_training(
             dpct::dnnl::batch_normalization_mode::per_activation,
             dpct::dnnl::batch_normalization_ops::activation, ActivationDesc,
             eps, factor, alpha, dataTensor, data, beta, outTensor, out,
             outTensor, nullptr, scalebiasTensor, scale, bias, rmean, rvar,
             smean, svar, reservespace_size, reservespace),
         0);
    float *bworkspace;
    size_t bworkspace_size;

    bworkspace_size = 0;

    bworkspace = (float *)sycl::malloc_device(bworkspace_size, q_ct1);

    handle.async_batch_normalization_backward(
        dpct::dnnl::batch_normalization_mode::per_activation,
        dpct::dnnl::batch_normalization_ops::activation, ActivationDesc, eps,
        alpha, dataTensor, data, outTensor, out, outTensor, diffout, beta,
        dataTensor, diffdata, outTensor, diffz, alpha, scalebiasTensor, scale,
        bias, beta, diffscale, diffbias, smean, svar, reservespace_size,
        reservespace);

    dev_ct1.queues_wait_and_throw();
    q_ct1.memcpy(host_out.data(), out, sizeof(float) * oele_num).wait();
    q_ct1.memcpy(host_smean.data(), smean, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_svar.data(), svar, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_rmean.data(), rmean, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_rvar.data(), rvar, sizeof(float) * save * sele_num)
        .wait();
    q_ct1.memcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num)
        .wait();
    q_ct1.memcpy(host_diffz.data(), diffz, sizeof(float) * ele_num).wait();
    q_ct1
        .memcpy(host_diffscale.data(), diffscale,
                sizeof(float) * save * sele_num)
        .wait();
    q_ct1
        .memcpy(host_diffbias.data(), diffbias, sizeof(float) * save * sele_num)
        .wait();

    std::vector<float> expect = {
        1.502, 1.5025, 1.503, 1.5035, 1.504,
        1.5045, 1.505, 1.5055, 1.506, 1.5065,
        1.507, 1.5075, 1.508, 1.5085, 1.509,
        1.5095, 1.51, 1.5105, 1.511, 1.5115,
        1.512, 1.5125, 1.513, 1.5135, 1.514,
        
        1.5145, 1.515, 1.5155, 1.516, 1.5165,
        1.517, 1.5175, 1.518, 1.5185, 1.519,
        1.5195, 1.52, 1.5205, 1.521, 1.5215,
        1.522, 1.5225, 1.523, 1.5235, 1.524,
        1.5245, 1.525, 1.5255, 1.526, 1.5265,
        
        1.527, 1.5275, 1.528, 1.5285, 1.529,
        1.5295, 1.53, 1.5305, 1.531, 1.5315,
        1.532, 1.5325, 1.533, 1.5335, 1.534,
        1.5345, 1.535, 1.5355, 1.536, 1.5365,
        1.537, 1.5375, 1.538, 1.5385, 1.539,
        
        1.5395, 1.54, 1.5405, 1.541, 1.5415,
        1.542, 1.5425, 1.543, 1.5435, 1.544,
        1.54449, 1.54499, 1.54549, 1.54599, 1.54649,
        1.54699, 1.54749, 1.54799, 1.54849, 1.54899,
        1.54949, 1.54999, 1.55049, 1.55099, 1.55149,

        21.498, 26.4975, 31.497, 36.4965, 41.496,
        46.4955, 51.495, 56.4945, 61.494, 66.4935,
        71.493, 76.4925, 81.492, 86.4915, 91.491,
        96.4905, 101.49, 106.49, 111.489, 116.488,
        121.488, 126.487, 131.487, 136.487, 141.486,
        
        146.486, 151.485, 156.484, 161.484, 166.484,
        171.483, 176.483, 181.482, 186.482, 191.481,
        196.48, 201.48, 206.479, 211.479, 216.479,
        221.478, 226.477, 231.477, 236.477, 241.476,
        246.476, 251.475, 256.474, 261.474, 266.474,
        
        271.473, 276.473, 281.472, 286.471, 291.471,
        296.471, 301.47, 306.47, 311.469, 316.469,
        321.468, 326.467, 331.467, 336.466, 341.466,
        346.466, 351.465, 356.464, 361.464, 366.464,
        371.463, 376.462, 381.462, 386.462, 391.461,
        
        396.461, 401.46, 406.459, 411.459, 416.458,
        421.458, 426.458, 431.457, 436.457, 441.456,
        446.456, 451.455, 456.454, 461.454, 466.453,
        471.453, 476.453, 481.452, 486.452, 491.451,
        496.451, 501.45, 506.449, 511.449, 516.448
        };

    std::vector<float> expect_diffdata = {
        -0.399822, -0.499778, -0.599734, -0.699689, -0.799645,
        -0.8996, -0.999556, -1.09951, -1.19947, -1.29942,
        -1.39938, -1.49933, -1.59929, -1.69925, -1.7992,
        -1.89916, -1.99911, -2.09907, -2.19902, -2.29898,
        -2.39893, -2.49889, -2.59885, -2.6988, -2.79876,
        
        -2.89871, -2.99867, -3.09862, -3.19858, -3.29853,
        -3.39849, -3.49845, -3.59928, -3.69926, -3.79924,
        -3.89922, -3.9992, -4.09918, -4.19916, -4.29914,
        -4.39912, -4.4991, -4.59908, -4.69906, -4.79904,
        -4.89902, -4.999, -5.09898, -5.19896, -5.29894,
        
        -5.39892, -5.4989, -5.59888, -5.69886, -5.79884,
        -5.89882, -5.9988, -6.09878, -6.19876, -6.29874,
        -6.39872, -6.4987, -6.59868, -6.69866, -6.79864,
        -6.89862, -6.9986, -7.09858, -7.19856, -7.29854,
        -7.39852, -7.4985, -7.59848, -7.69846, -7.79844,
        
        -7.89842, -7.9984, -8.09838, -8.19836, -8.29834,
        -8.39832, -8.4983, -8.59828, -8.69826, -8.79824,
        -8.89822, -8.9982, -9.09818, -9.19816, -9.29814,
        -9.39812, -9.4981, -9.59808, -9.69806, -9.79804,
        -9.89802, -9.998, -10.098, -10.198, -10.2979,

        0.399822, 0.499778, 0.599734, 0.699689, 0.799645,
        0.8996, 0.999556, 1.09951, 1.19947, 1.29942,
        1.39938, 1.49933, 1.59929, 1.69925, 1.7992,
        1.89916, 1.99911, 2.09907, 2.19902, 2.29898,
        2.39893, 2.49889, 2.59885, 2.6988, 2.79876,
        
        2.89871, 2.99867, 3.09862, 3.19858, 3.29853,
        3.39849, 3.49845, 3.59928, 3.69926, 3.79924,
        3.89922, 3.9992, 4.09918, 4.19916, 4.29914,
        4.39912, 4.4991, 4.59908, 4.69906, 4.79904,
        4.89902, 4.999, 5.09898, 5.19896, 5.29894,
        
        5.39892, 5.4989, 5.59888, 5.69886, 5.79884,
        5.89882, 5.9988, 6.09878, 6.19876, 6.29874,
        6.39872, 6.4987, 6.59868, 6.69866, 6.79864,
        6.89862, 6.9986, 7.09858, 7.19856, 7.29854,
        7.39852, 7.4985, 7.59848, 7.69846, 7.79844,
        
        7.89842, 7.9984, 8.09838, 8.19836, 8.29834,
        8.39832, 8.4983, 8.59828, 8.69826, 8.79824,
        8.89822, 8.9982, 9.09818, 9.19816, 9.29814,
        9.39812, 9.4981, 9.59808, 9.69806, 9.79804,
        9.89802, 9.998, 10.098, 10.198, 10.2979                     
        };
    std::vector<float> expect_diffscale = {
        25001, 25002.5, 25004, 25005.5, 25007,
        25008.5, 25010, 25011.5, 25013, 25014.5,
        25016, 25017.5, 25019, 25020.5, 25022,
        25023.5, 25025, 25026.5, 25028, 25029.5,
        25031, 25032.5, 25034, 25035.5, 25037,
        
        25038.5, 25040, 25041.5, 25043, 25044.5,
        25046, 25047.5, 25049, 25050.5, 25052,
        25053.5, 25055, 25056.5, 25058, 25059.5,
        25061, 25062.5, 25064, 25065.5, 25067,
        25068.5, 25070, 25071.5, 25073, 25074.5,
        
        25076, 25077.5, 25079, 25080.5, 25082,
        25083.5, 25085, 25086.5, 25088, 25089.5,
        25091, 25092.5, 25094, 25095.5, 25097,
        25098.5, 25100, 25101.5, 25103, 25104.5,
        25106, 25107.5, 25109, 25110.5, 25112,
        
        25113.5, 25115, 25116.5, 25118, 25119.5,
        25121, 25122.5, 25124, 25125.5, 25127,
        25128.5, 25130, 25131.5, 25133, 25134.5,
        25136, 25137.5, 25139, 25140.5, 25142,
        25143.5, 25145, 25146.5, 25148, 25149.5             
        };

    std::vector<float> expect_diffbias = {
        25011, 25512.5, 26014, 26515.5, 27017,
        27518.5, 28020, 28521.5, 29023, 29524.5,
        30026, 30527.5, 31029, 31530.5, 32032,
        32533.5, 33035, 33536.5, 34038, 34539.5,
        35041, 35542.5, 36044, 36545.5, 37047,
        
        37548.5, 38050, 38551.5, 39053, 39554.5,
        40056, 40557.5, 41059, 41560.5, 42062,
        42563.5, 43065, 43566.5, 44068, 44569.5,
        45071, 45572.5, 46074, 46575.5, 47077,
        47578.5, 48080, 48581.5, 49083, 49584.5,
        
        50086, 50587.5, 51089, 51590.5, 52092,
        52593.5, 53095, 53596.5, 54098, 54599.5,
        55101, 55602.5, 56104, 56605.5, 57107,
        57608.5, 58110, 58611.5, 59113, 59614.5,
        60116, 60617.5, 61119, 61620.5, 62122,
        
        62623.5, 63125, 63626.5, 64128, 64629.5,
        65131, 65632.5, 66134, 66635.5, 67137,
        67638.5, 68140, 68641.5, 69143, 69644.5,
        70146, 70647.5, 71149, 71650.5, 72152,
        72653.5, 73155, 73656.5, 74158, 74659.5,        
        };
    check(expect, host_out, expect.size(), 1e-1);
    check(expect_diffdata, host_diffdata, expect_diffdata.size(), 1e-1);
    check(expect_diffscale, host_diffscale, expect_diffscale.size(), 1e-1);
    check(expect_diffbias, host_diffbias, expect_diffbias.size(), 1e-1);
        sycl::free(data, q_ct1);
    sycl::free(out, q_ct1);
}

int main() {
    test7<dpct::library_data_t::real_float>();
    test8<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}