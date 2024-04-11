// ===------- profiler.cu ------------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#include <cuda_profiler_api.h>
int main() {
  cudaProfilerStart();
  cudaError_t result = cudaProfilerStart();

  cudaProfilerStop();
  cudaError_t r2 = cudaProfilerStop();
  return 0;
}
