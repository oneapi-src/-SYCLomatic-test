// ====------ dnnl_utils_rnn.cpp--------------===////
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
void check(std::vector<T> &expect, T *actual, int num, float precision) {
  for(int i = 0; i < num; i++){
      if(std::abs(expect[i] - actual[i]) > precision) {
          std::cout << "test failed" << std::endl;
          std::cout << "expect:" << expect[i] << std::endl;
          std::cout << "actual:" << actual[i] << std::endl;
          exit(-1);
      }
  }
}

template<typename T = float>
void initData(T *data, T init, int size, bool inc = false){
  for(int i = 0; i < size; i++) {
      data[i] = init + (inc ? 1.f : 0.f);
  }
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test1() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 1;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    dpct::dnnl::engine_ext handle;
    handle.create_engine();

    dpct::dnnl::rnn_desc rnnDesc;
    dpct::dnnl::memory_desc_ext xDesc;
    dpct::dnnl::memory_desc_ext yDesc;
    dpct::dnnl::memory_desc_ext hDesc;
    dpct::dnnl::memory_desc_ext cDesc;

    size_t spacesize, statesize;
    void* reservespace, *state;

    hDesc.set(dpct::library_data_t::real_float, 3, hDim, hStride);
    cDesc.set(dpct::library_data_t::real_float, 3, cDim, cStride);

    rnnDesc.set(dpct::dnnl::rnn_mode::vanilla_relu,
                dpct::dnnl::rnn_bias_mode::single,
                dir == 1 ? dpct::dnnl::rnn_direction::unidirectional
                         : dpct::dnnl::rnn_direction::bidirectional,
                dpct::library_data_t::real_float, inputSize, hidenSize,
                projectSize, layerSize);

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    xDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, xDim[0], xDim[1], xDim[2]);

    yDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, yDim[0], yDim[1], yDim[2]);
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;

    auto ss = (handle.rnn_get_weight_space_size(rnnDesc, &weightsSpaceSize), 0);
    handle.rnn_get_scratchpad_workspace_size(
        rnnDesc, dnnl::prop_kind::forward_training, xDesc, &workSpaceSize,
        &reserveSpaceSize);

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    xData = sycl::malloc_device<float>(x_size, q_ct1);
    yData = sycl::malloc_device<float>(y_size, q_ct1);
    hxData = sycl::malloc_device<float>(h_size, q_ct1);
    hyData = sycl::malloc_device<float>(h_size, q_ct1);
    cxData = sycl::malloc_device<float>(c_size, q_ct1);
    cyData = sycl::malloc_device<float>(c_size, q_ct1);
    weightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);
    workSpaceData = (float *)sycl::malloc_device(workSpaceSize, q_ct1);
    reserveSpaceData = (float *)sycl::malloc_device(reserveSpaceSize, q_ct1);

    dxData = sycl::malloc_device<float>(x_size, q_ct1);
    dyData = sycl::malloc_device<float>(y_size, q_ct1);
    dhxData = sycl::malloc_device<float>(h_size, q_ct1);
    dhyData = sycl::malloc_device<float>(h_size, q_ct1);
    dcxData = sycl::malloc_device<float>(c_size, q_ct1);
    dcyData = sycl::malloc_device<float>(c_size, q_ct1);
    dweightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    q_ct1.memcpy(xData, host_xData, sizeof(float) * x_size);
    q_ct1.memcpy(yData, host_yData, sizeof(float) * y_size);
    q_ct1.memcpy(hxData, host_hxData, sizeof(float) * h_size);
    q_ct1.memcpy(hyData, host_hyData, sizeof(float) * h_size);
    q_ct1.memcpy(cxData, host_cxData, sizeof(float) * c_size);
    q_ct1.memcpy(cyData, host_cyData, sizeof(float) * c_size);
    q_ct1.memcpy(weightsData, host_weightsData, weightsSpaceSize);
    q_ct1.memcpy(workSpaceData, host_workSpaceData, workSpaceSize);
    q_ct1.memcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize);

    q_ct1.memcpy(dxData, host_dxData, sizeof(float) * x_size);
    q_ct1.memcpy(dyData, host_dyData, sizeof(float) * y_size);
    q_ct1.memcpy(dhxData, host_dhxData, sizeof(float) * h_size);
    q_ct1.memcpy(dhyData, host_dhyData, sizeof(float) * h_size);
    q_ct1.memcpy(dcxData, host_dcxData, sizeof(float) * c_size);
    q_ct1.memcpy(dcyData, host_dcyData, sizeof(float) * c_size);
    q_ct1.memcpy(dweightsData, host_dweightsData, weightsSpaceSize).wait();

    int *seqlenarray;
    seqlenarray = sycl::malloc_device<int>(3, q_ct1);
    q_ct1.memcpy(seqlenarray, seqLenArray, sizeof(int) * 3).wait();

    auto e = (handle.async_rnn_forward(
                  rnnDesc, dnnl::prop_kind::forward_training, xDesc, xData,
                  yDesc, yData, hDesc, hxData, hyData, cDesc, cxData, cyData,
                  weightsSpaceSize, weightsData, workSpaceSize, workSpaceData,
                  reserveSpaceSize, reserveSpaceData),
              0);

    q_ct1.memcpy(host_yData, yData, sizeof(float) * y_size).wait();

    std::vector<float> expect = {
        29, 29,
        29, 29,
        29, 29,   
        165, 165,
        165, 165,
        165, 165,
        661, 661,
        661, 661,
        661, 661,
        2229, 2229,
        2229, 2229,
        2229, 2229,
      };

    check(expect, host_yData, expect.size(), 1e-3);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 2;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    dpct::dnnl::engine_ext handle;
    handle.create_engine();

    dpct::dnnl::rnn_desc rnnDesc;
    dpct::dnnl::memory_desc_ext xDesc;
    dpct::dnnl::memory_desc_ext yDesc;
    dpct::dnnl::memory_desc_ext hDesc;
    dpct::dnnl::memory_desc_ext cDesc;

    size_t spacesize, statesize;
    void* reservespace, *state;

    hDesc.set(dpct::library_data_t::real_float, 3, hDim, hStride);
    cDesc.set(dpct::library_data_t::real_float, 3, cDim, cStride);

    rnnDesc.set(dpct::dnnl::rnn_mode::vanilla_relu,
                dpct::dnnl::rnn_bias_mode::single,
                dir == 1 ? dpct::dnnl::rnn_direction::unidirectional
                         : dpct::dnnl::rnn_direction::bidirectional,
                dpct::library_data_t::real_float, inputSize, hidenSize,
                projectSize, layerSize);

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    xDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, xDim[0], xDim[1], xDim[2]);

    yDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, yDim[0], yDim[1], yDim[2]);
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;

    auto ss = (handle.rnn_get_weight_space_size(rnnDesc, &weightsSpaceSize), 0);
    handle.rnn_get_scratchpad_workspace_size(
        rnnDesc, dnnl::prop_kind::forward_training, xDesc, &workSpaceSize,
        &reserveSpaceSize);

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    xData = sycl::malloc_device<float>(x_size, q_ct1);
    yData = sycl::malloc_device<float>(y_size, q_ct1);
    hxData = sycl::malloc_device<float>(h_size, q_ct1);
    hyData = sycl::malloc_device<float>(h_size, q_ct1);
    cxData = sycl::malloc_device<float>(c_size, q_ct1);
    cyData = sycl::malloc_device<float>(c_size, q_ct1);
    weightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);
    workSpaceData = (float *)sycl::malloc_device(workSpaceSize, q_ct1);
    reserveSpaceData = (float *)sycl::malloc_device(reserveSpaceSize, q_ct1);

    dxData = sycl::malloc_device<float>(x_size, q_ct1);
    dyData = sycl::malloc_device<float>(y_size, q_ct1);
    dhxData = sycl::malloc_device<float>(h_size, q_ct1);
    dhyData = sycl::malloc_device<float>(h_size, q_ct1);
    dcxData = sycl::malloc_device<float>(c_size, q_ct1);
    dcyData = sycl::malloc_device<float>(c_size, q_ct1);
    dweightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    q_ct1.memcpy(xData, host_xData, sizeof(float) * x_size);
    q_ct1.memcpy(yData, host_yData, sizeof(float) * y_size);
    q_ct1.memcpy(hxData, host_hxData, sizeof(float) * h_size);
    q_ct1.memcpy(hyData, host_hyData, sizeof(float) * h_size);
    q_ct1.memcpy(cxData, host_cxData, sizeof(float) * c_size);
    q_ct1.memcpy(cyData, host_cyData, sizeof(float) * c_size);
    q_ct1.memcpy(weightsData, host_weightsData, weightsSpaceSize);
    q_ct1.memcpy(workSpaceData, host_workSpaceData, workSpaceSize);
    q_ct1.memcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize);

    q_ct1.memcpy(dxData, host_dxData, sizeof(float) * x_size);
    q_ct1.memcpy(dyData, host_dyData, sizeof(float) * y_size);
    q_ct1.memcpy(dhxData, host_dhxData, sizeof(float) * h_size);
    q_ct1.memcpy(dhyData, host_dhyData, sizeof(float) * h_size);
    q_ct1.memcpy(dcxData, host_dcxData, sizeof(float) * c_size);
    q_ct1.memcpy(dcyData, host_dcyData, sizeof(float) * c_size);
    q_ct1.memcpy(dweightsData, host_dweightsData, weightsSpaceSize).wait();

    int *seqlenarray;
    seqlenarray = sycl::malloc_device<int>(3, q_ct1);
    q_ct1.memcpy(seqlenarray, seqLenArray, sizeof(int) * 3).wait();

    auto e = (handle.async_rnn_forward(
                  rnnDesc, dnnl::prop_kind::forward_training, xDesc, xData,
                  yDesc, yData, hDesc, hxData, hyData, cDesc, cxData, cyData,
                  weightsSpaceSize, weightsData, workSpaceSize, workSpaceData,
                  reserveSpaceSize, reserveSpaceData),
              0);

    q_ct1.memcpy(host_yData, yData, sizeof(float) * y_size).wait();

    std::vector<float> expect = {
        3719, 3719, 47275, 47275,
        3719, 3719, 47275, 47275,
        3719, 3719, 47275, 47275,

        9739, 9739, 21779, 21779,
        9739, 9739, 21779, 21779,
        9739, 9739, 21779, 21779,
        
        21779, 21779, 9739, 9739,
        21779, 21779, 9739, 9739,
        21779, 21779, 9739, 9739,
        
        47275, 47275, 3719, 3719,
        47275, 47275, 3719, 3719,
        47275, 47275, 3719, 3719,
      };

    check(expect, host_yData, expect.size(), 1e-3);
}

template <dpct::library_data_t T, typename HT = typename dt_trait<T>::type>
void test3() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 1;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    dpct::dnnl::engine_ext handle;
    handle.create_engine();

    dpct::dnnl::rnn_desc rnnDesc;
    dpct::dnnl::memory_desc_ext xDesc;
    dpct::dnnl::memory_desc_ext yDesc;
    dpct::dnnl::memory_desc_ext hDesc;
    dpct::dnnl::memory_desc_ext cDesc;

    size_t spacesize, statesize;
    void* reservespace, *state;

    hDesc.set(dpct::library_data_t::real_float, 3, hDim, hStride);
    cDesc.set(dpct::library_data_t::real_float, 3, cDim, cStride);

    rnnDesc.set(dpct::dnnl::rnn_mode::vanilla_relu,
                dpct::dnnl::rnn_bias_mode::single,
                dir == 1 ? dpct::dnnl::rnn_direction::unidirectional
                         : dpct::dnnl::rnn_direction::bidirectional,
                dpct::library_data_t::real_float, inputSize, hidenSize,
                projectSize, layerSize);

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    xDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, xDim[0], xDim[1], xDim[2]);

    yDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc,
              dpct::library_data_t::real_float, yDim[0], yDim[1], yDim[2]);
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;

    auto ss = (handle.rnn_get_weight_space_size(rnnDesc, &weightsSpaceSize), 0);
    handle.rnn_get_scratchpad_workspace_size(
        rnnDesc, dnnl::prop_kind::forward_training, xDesc, &workSpaceSize,
        &reserveSpaceSize);

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    xData = sycl::malloc_device<float>(x_size, q_ct1);
    yData = sycl::malloc_device<float>(y_size, q_ct1);
    hxData = sycl::malloc_device<float>(h_size, q_ct1);
    hyData = sycl::malloc_device<float>(h_size, q_ct1);
    cxData = sycl::malloc_device<float>(c_size, q_ct1);
    cyData = sycl::malloc_device<float>(c_size, q_ct1);
    weightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);
    workSpaceData = (float *)sycl::malloc_device(workSpaceSize, q_ct1);
    reserveSpaceData = (float *)sycl::malloc_device(reserveSpaceSize, q_ct1);

    dxData = sycl::malloc_device<float>(x_size, q_ct1);
    dyData = sycl::malloc_device<float>(y_size, q_ct1);
    dhxData = sycl::malloc_device<float>(h_size, q_ct1);
    dhyData = sycl::malloc_device<float>(h_size, q_ct1);
    dcxData = sycl::malloc_device<float>(c_size, q_ct1);
    dcyData = sycl::malloc_device<float>(c_size, q_ct1);
    dweightsData = (float *)sycl::malloc_device(weightsSpaceSize, q_ct1);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    q_ct1.memcpy(xData, host_xData, sizeof(float) * x_size);
    q_ct1.memcpy(yData, host_yData, sizeof(float) * y_size);
    q_ct1.memcpy(hxData, host_hxData, sizeof(float) * h_size);
    q_ct1.memcpy(hyData, host_hyData, sizeof(float) * h_size);
    q_ct1.memcpy(cxData, host_cxData, sizeof(float) * c_size);
    q_ct1.memcpy(cyData, host_cyData, sizeof(float) * c_size);
    q_ct1.memcpy(weightsData, host_weightsData, weightsSpaceSize);
    q_ct1.memcpy(workSpaceData, host_workSpaceData, workSpaceSize);
    q_ct1.memcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize);

    q_ct1.memcpy(dxData, host_dxData, sizeof(float) * x_size);
    q_ct1.memcpy(dyData, host_dyData, sizeof(float) * y_size);
    q_ct1.memcpy(dhxData, host_dhxData, sizeof(float) * h_size);
    q_ct1.memcpy(dhyData, host_dhyData, sizeof(float) * h_size);
    q_ct1.memcpy(dcxData, host_dcxData, sizeof(float) * c_size);
    q_ct1.memcpy(dcyData, host_dcyData, sizeof(float) * c_size);
    q_ct1.memcpy(dweightsData, host_dweightsData, weightsSpaceSize).wait();

    int *seqlenarray;
    seqlenarray = sycl::malloc_device<int>(3, q_ct1);
    q_ct1.memcpy(seqlenarray, seqLenArray, sizeof(int) * 3).wait();

    auto e = (handle.async_rnn_forward(
                  rnnDesc, dnnl::prop_kind::forward_training, xDesc, xData,
                  yDesc, yData, hDesc, hxData, hyData, cDesc, cxData, cyData,
                  weightsSpaceSize, weightsData, workSpaceSize, workSpaceData,
                  reserveSpaceSize, reserveSpaceData),
              0);

    q_ct1.memcpy(host_yData, yData, sizeof(float) * y_size).wait();

    std::vector<float> expect = {
        29, 29,
        29, 29,
        29, 29,   
        165, 165,
        165, 165,
        165, 165,
        661, 661,
        661, 661,
        661, 661,
        2229, 2229,
        2229, 2229,
        2229, 2229,
      };

    check(expect, host_yData, expect.size(), 1e-3);

    handle.async_rnn_backward(
        rnnDesc,
        yDesc,
        yData,
        dyData,
        xDesc,
        xData,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        dweightsData,
        workSpaceSize,
        workSpaceData,
        reserveSpaceSize,
        reserveSpaceData
    );

    q_ct1.memcpy(host_dxData, dxData,
                 sizeof(float) * x_size).wait();
    q_ct1.memcpy(host_dhxData, dhxData,
                 sizeof(float) * h_size).wait();
    std::vector<float> expect_dx = {
        1672, 1672,
        1672, 1672,
        1672, 1672,
        
        496, 496,
        496, 496,
        496, 496,
        
        124, 124,
        124, 124,
        124, 124,
        
        22, 22,
        22, 22,
        22, 22,
      };
    std::vector<float> expect_dhx = {
        1672, 1672,
        1672, 1672,
        1672, 1672,
        
        340, 340,
        340, 340,
        340, 340,
        
        46, 46,
        46, 46,
        46, 46,
      };
      check(expect_dx, host_dxData, expect_dx.size(), 1e-3);
      check(expect_dhx, host_dhxData, expect_dhx.size(), 1e-3);
}

int main() {
    test1<dpct::library_data_t::real_float>();
    test2<dpct::library_data_t::real_float>();
    test3<dpct::library_data_t::real_float>();
    std::cout << "test passed" << std::endl;
    return 0;
}