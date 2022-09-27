// ====------ cudnn-normp3.cu ---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};

template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
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


template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test7() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&scalebiasTensor);

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);


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
        host_z[i] = 10;
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
        host_diffscale[i] = i + 4.f;
        host_diffbias[i] = i + 4.f;
    }

    cudaMalloc(&data, sizeof(float) * ele_num);
    cudaMalloc(&z, sizeof(float) * oele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);
    cudaMalloc(&diffout, sizeof(float) * oele_num);
    cudaMalloc(&diffz, sizeof(float) * oele_num);
    cudaMalloc(&diffdata, sizeof(float) * ele_num);
    cudaMalloc(&diffscale, sizeof(float) * sele_num);
    cudaMalloc(&diffbias, sizeof(float) * sele_num);

    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * oele_num,
      cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(diffscale, host_diffscale.data(), sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(diffbias, host_diffbias.data(), sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    float alpha = 2.5f, beta = 0.f, eps = 1.f;
    double factor = 0.5f;
    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    float *workspace, *reservespace;
    size_t workspace_size, reservespace_size;

    cudnnGetNormalizationForwardTrainingWorkspaceSize(
        handle, 
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        dataTensor,
        outTensor,
        scalebiasTensor,
        ActivationDesc,
        scalebiasTensor,
        &workspace_size,
        1
    );
    cudnnGetNormalizationTrainingReserveSpaceSize(
        handle,
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        NULL,
        dataTensor,
        &reservespace_size,
        1
    );
    cudaMalloc(&workspace, workspace_size);
    cudaMalloc(&reservespace,  reservespace_size);
    auto status = cudnnNormalizationForwardTraining(
        handle, 
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        dataTensor,
        data,
        scalebiasTensor,
        scale,
        bias,
        factor,
        scalebiasTensor,
        rmean,
        rvar,
        eps,
        smean,
        svar,
        ActivationDesc,
        dataTensor,
        z,
        outTensor,
        out,
        workspace,
        workspace_size,
        reservespace,
        reservespace_size,
        1);


    float *bworkspace;
    size_t bworkspace_size;

    cudnnGetNormalizationBackwardWorkspaceSize(
        handle, 
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        outTensor,
        outTensor,
        outTensor,
        dataTensor,
        scalebiasTensor,
        ActivationDesc,
        scalebiasTensor,
        &bworkspace_size,
        1
    );
    cudaMalloc(&bworkspace, bworkspace_size);
    status = cudnnNormalizationBackward(
        handle, 
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        out,
        outTensor,
        diffout,
        outTensor,
        diffz,
        dataTensor,
        diffdata,
        scalebiasTensor,
        scale,
        bias,
        diffscale,
        diffbias,
        eps,
        scalebiasTensor,
        smean,
        svar,
        ActivationDesc,
        bworkspace,
        bworkspace_size,
        reservespace,
        reservespace_size,
        1);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_smean.data(), smean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_svar.data(), svar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rmean.data(), rmean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rvar.data(), rvar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffz.data(), diffz, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffscale.data(), diffscale,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffbias.data(), diffbias,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0, -0.0999556, -0.199911, -0.299867, -0.399822,
        -0.499778, -0.599734, -0.699689, -0.799645, -0.8996,
        -0.999556, -1.09951, -1.19947, -1.29942, -1.39938,
        -1.49933, -1.59929, -1.69925, -1.7992, -1.89916,
        -1.99911, -2.09907, -2.19902, -2.29898, -2.39893,
        
        -2.49889, -2.59885, -2.6988, -2.79876, -2.89871,
        -2.99867, -3.09862, -3.19936, -3.29934, -3.39932,
        -3.4993, -3.59928, -3.69926, -3.79924, -3.89922,
        -3.9992, -4.09918, -4.19916, -4.29914, -4.39912,
        -4.4991, -4.59908, -4.69906, -4.79904, -4.89902,
        
        -4.999, -5.09898, -5.19896, -5.29894, -5.39892,
        -5.4989, -5.59888, -5.69886, -5.79884, -5.89882,
        -5.9988, -6.09878, -6.19876, -6.29874, -6.39872,
        -6.4987, -6.59868, -6.69866, -6.79864, -6.89862,
        -6.9986, -7.09858, -7.19856, -7.29854, -7.39852,
        
        -7.4985, -7.59848, -7.69846, -7.79844, -7.89842,
        -7.9984, -8.09838, -8.19836, -8.29834, -8.39832,
        -8.4983, -8.59828, -8.69826, -8.79824, -8.89822,
        -8.9982, -9.09818, -9.19816, -9.29814, -9.39812,
        -9.4981, -9.59808, -9.69806, -9.79804, -9.89802,

        0, 0.0999556, 0.199911, 0.299867, 0.399822,
        0.499778, 0.599734, 0.699689, 0.799645, 0.8996,
        0.999556, 1.09951, 1.19947, 1.29942, 1.39938,
        1.49933, 1.59929, 1.69925, 1.7992, 1.89916,
        1.99911, 2.09907, 2.19902, 2.29898, 2.39893,
        
        2.49889, 2.59885, 2.6988, 2.79876, 2.89871,
        2.99867, 3.09862, 3.19936, 3.29934, 3.39932,
        3.4993, 3.59928, 3.69926, 3.79924, 3.89922,
        3.9992, 4.09918, 4.19916, 4.29914, 4.39912,
        4.4991, 4.59908, 4.69906, 4.79904, 4.89902,
        
        4.999, 5.09898, 5.19896, 5.29894, 5.39892,
        5.4989, 5.59888, 5.69886, 5.79884, 5.89882,
        5.9988, 6.09878, 6.19876, 6.29874, 6.39872,
        6.4987, 6.59868, 6.69866, 6.79864, 6.89862,
        6.9986, 7.09858, 7.19856, 7.29854, 7.39852,
        
        7.4985, 7.59848, 7.69846, 7.79844, 7.89842,
        7.9984, 8.09838, 8.19836, 8.29834, 8.39832,
        8.4983, 8.59828, 8.69826, 8.79824, 8.89822,
        8.9982, 9.09818, 9.19816, 9.29814, 9.39812,
        9.4981, 9.59808, 9.69806, 9.79804, 9.89802,
        };

    std::vector<float> expect_diffscale = {
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,
        24995, 24995, 24995, 24995, 24995,                 
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
    check(expect_diffscale, host_diffscale, expect_diffscale.size(), 1.f);
    check(expect_diffbias, host_diffbias, expect_diffbias.size(), 1.f);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test8() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&scalebiasTensor);

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 1, sbw = 1;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

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
        host_z[i] = 10;
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
        host_diffscale[i] = i + 4.f;
        host_diffbias[i] = i + 4.f;
    }

    cudaMalloc(&data, sizeof(float) * ele_num);
    cudaMalloc(&z, sizeof(float) * oele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);
    cudaMalloc(&diffout, sizeof(float) * oele_num);
    cudaMalloc(&diffz, sizeof(float) * oele_num);
    cudaMalloc(&diffdata, sizeof(float) * ele_num);
    cudaMalloc(&diffscale, sizeof(float) * sele_num);
    cudaMalloc(&diffbias, sizeof(float) * sele_num);

    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * oele_num,
      cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(diffscale, host_diffscale.data(), sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(diffbias, host_diffbias.data(), sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    float alpha = 2.5f, beta = 0.f, eps = 1.f;
    double factor = 0.5f;
    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    float *workspace, *reservespace;
    size_t workspace_size, reservespace_size;

    cudnnGetNormalizationForwardTrainingWorkspaceSize(
        handle, 
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        dataTensor,
        outTensor,
        scalebiasTensor,
        ActivationDesc,
        scalebiasTensor,
        &workspace_size,
        1
    );
    cudnnGetNormalizationTrainingReserveSpaceSize(
        handle,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        NULL,
        dataTensor,
        &reservespace_size,
        1
    );
    cudaMalloc(&workspace, workspace_size);
    cudaMalloc(&reservespace,  reservespace_size);
    auto status = cudnnNormalizationForwardTraining(
        handle, 
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        dataTensor,
        data,
        scalebiasTensor,
        scale,
        bias,
        factor,
        scalebiasTensor,
        rmean,
        rvar,
        eps,
        smean,
        svar,
        ActivationDesc,
        dataTensor,
        z,
        outTensor,
        out,
        workspace,
        workspace_size,
        reservespace,
        reservespace_size,
        1);


    float *bworkspace;
    size_t bworkspace_size;

    cudnnGetNormalizationBackwardWorkspaceSize(
        handle, 
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        outTensor,
        outTensor,
        outTensor,
        dataTensor,
        scalebiasTensor,
        ActivationDesc,
        scalebiasTensor,
        &bworkspace_size,
        1
    );
    cudaMalloc(&bworkspace, bworkspace_size);
    status = cudnnNormalizationBackward(
        handle, 
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        out,
        outTensor,
        diffout,
        outTensor,
        diffz,
        dataTensor,
        diffdata,
        scalebiasTensor,
        scale,
        bias,
        diffscale,
        diffbias,
        eps,
        scalebiasTensor,
        smean,
        svar,
        ActivationDesc,
        bworkspace,
        bworkspace_size,
        reservespace,
        reservespace_size,
        1);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_smean.data(), smean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_svar.data(), svar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rmean.data(), rmean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rvar.data(), rvar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffz.data(), diffz, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffscale.data(), diffscale,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffbias.data(), diffbias,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        
        -0.120178, -0.118256, -0.116302, -0.11438, -0.112427,
        -0.110474, -0.108551, -0.106598, -0.104675, -0.102722,
        -0.1008, -0.0988464, -0.0969238, -0.0950012, -0.0930481,
        -0.091095, -0.0891724, -0.0872192, -0.0852966, -0.0833435,
        -0.0814209, -0.0794678, -0.0775452, -0.075592, -0.0736694,
        
        -0.240356, -0.23645, -0.232544, -0.22876, -0.224854,
        -0.220947, -0.217041, -0.213135, -0.209229, -0.205444,
        -0.201538, -0.197632, -0.193848, -0.189941, -0.186035,
        -0.182129, -0.178223, -0.174438, -0.170532, -0.166626,
        -0.162842, -0.158936, -0.155029, -0.151123, -0.147339,
        
        -0.360474, -0.354614, -0.348877, -0.343018, -0.337158,
        -0.331421, -0.325562, -0.319824, -0.313965, -0.308105,
        -0.302368, -0.296509, -0.290649, -0.284912, -0.279053,
        -0.273315, -0.267456, -0.261597, -0.255859, -0.25,
        -0.244141, -0.238403, -0.232544, -0.226685, -0.220947,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        
        0.0736389, 0.0755615, 0.0775146, 0.0794373, 0.0813904,
        0.0833435, 0.0852661, 0.0872192, 0.0891418, 0.0910645,
        0.0930176, 0.0949707, 0.0968933, 0.0988464, 0.100769,
        0.102692, 0.104645, 0.106567, 0.108521, 0.110443,
        0.112427, 0.114349, 0.116302, 0.118225, 0.120148,
        
        0.147339, 0.151245, 0.155029, 0.158936, 0.162842,
        0.166748, 0.170532, 0.174438, 0.178345, 0.182251,
        0.186157, 0.189941, 0.193848, 0.197754, 0.201538,
        0.205444, 0.209351, 0.213257, 0.217041, 0.220947,
        0.224854, 0.22876, 0.232666, 0.23645, 0.240356,
        
        0.220947, 0.226807, 0.232544, 0.238403, 0.244263,
        0.25, 0.255859, 0.261719, 0.267456, 0.273315,
        0.279175, 0.284912, 0.290771, 0.296631, 0.302368,
        0.308228, 0.313965, 0.319824, 0.325684, 0.331421,
        0.33728, 0.34314, 0.348877, 0.354736, 0.360596,
        };

    std::vector<float> expect_diffscale = {
        631343,

        631343,
        
        631343,
        
        631343,                        
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
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    test7<CUDNN_DATA_FLOAT>();
    test8<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}