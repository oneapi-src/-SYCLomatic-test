#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template <typename alg> void check(int &count, alg &perf) {
  if ((count != 1) || (perf.algo != CUDNN_CONVOLUTION_FWD_ALGO_GEMM)) {
    std::cout << "test failed" << std::endl;
    exit(-1);
  }
  count = 0;
}

int main() {
  cudnnHandle_t handle;
  cudnnConvolutionDescriptor_t covdes;
  cudnnTensorDescriptor_t dataTensor, outTensor;
  cudnnFilterDescriptor_t filterTensor;
  cudnnCreate(&handle);

  int returned_count = 0;

  cudnnConvolutionFwdAlgoPerf_t fwd_perf;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;

  cudnnFindConvolutionForwardAlgorithm(handle, dataTensor, filterTensor, covdes,
                                       outTensor, 1, &returned_count,
                                       &fwd_perf);
  check(returned_count, fwd_perf);

  cudnnFindConvolutionBackwardDataAlgorithm(handle, filterTensor, outTensor,
                                            covdes, dataTensor, 1,
                                            &returned_count, &bwd_data_perf);
  check(returned_count, bwd_data_perf);

  cudnnFindConvolutionBackwardFilterAlgorithm(
      handle, dataTensor, outTensor, covdes, filterTensor, 1, &returned_count,
      &bwd_filter_perf);
  check(returned_count, bwd_filter_perf);

  cudnnConvolutionFwdAlgoPerf_t fwd_perf1;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf1;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf1;

  cudnnGetConvolutionForwardAlgorithm_v7(handle, dataTensor, filterTensor,
                                         covdes, outTensor, 1, &returned_count,
                                         &fwd_perf1);
  check(returned_count, fwd_perf1);

  cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      handle, dataTensor, outTensor, covdes, filterTensor, 1, &returned_count,
      &bwd_filter_perf1);
  check(returned_count, bwd_filter_perf1);

  cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterTensor, outTensor,
                                              covdes, dataTensor, 1,
                                              &returned_count, &bwd_data_perf1);
  check(returned_count, bwd_data_perf1);

  std::cout << "test passed" << std::endl;
  return 0;
}