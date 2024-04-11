// ===------- deviceAPI.cu ------------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

int main() {
  cudaSetDeviceFlags(cudaDeviceMapHost);
  float *d_dst;
  cudaStream_t stream;
  // unsupported API
  // cudaMemcpyPeerAsync(d_dst, 1, d_dst, 1, 111, stream);

  int concurrentManagedAccess = 0;
  int p_gpuDevice = 0;
  cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, p_gpuDevice);
  int alignment;
  cudaDeviceGetAttribute(&alignment, cudaDevAttrTextureAlignment, p_gpuDevice);
  return 0;
}
