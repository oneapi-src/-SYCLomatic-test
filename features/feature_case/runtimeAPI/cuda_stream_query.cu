// ====------ cuda_stream_query.cu---------- *- CUDA -* ----------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <iostream>

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *d_data;
  cudaMalloc(&d_data, sizeof(float));
  cudaMemsetAsync(d_data, 0, sizeof(float), stream);

  cudaError_t status = cudaStreamQuery(stream);
  if (status == cudaSuccess) {
    std::cout << "Stream operations have completed." << std::endl;
  } else if (status == cudaErrorNotReady) {
    std::cout << "Stream operations are still in progress." << std::endl;
  } else {
    std::cerr << "An error occurred while querying the stream status."
              << std::endl;
  }
  cudaFree(d_data);
  cudaStreamDestroy(stream);

  return 0;
}
