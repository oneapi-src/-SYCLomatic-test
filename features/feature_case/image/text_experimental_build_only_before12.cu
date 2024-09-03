// ===----- text_experimental_build_only_before12.cu ----- *- CUDA -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>

static texture<float4, 2> r;

void Runtime_MemoryManagement() {
  cudaArray_t a = nullptr;
  size_t s = 1;
  cudaMemcpyKind k = cudaMemcpyDefault;
  void *v = nullptr;
  cudaMemcpyArrayToArray(a, s, s, a, s, s, s, k);
  cudaMemcpyFromArray(v, a, s, s, s, k);
  cudaMemcpyFromArrayAsync(v, a, s, s, s, k);
  cudaMemcpyToArray(a, s, s, v, s, k);
  cudaMemcpyToArrayAsync(a, s, s, v, s, k);
}

void Runtime_TextureReferenceManagement() {
  size_t s = 1;
  void *v;
  cudaChannelFormatDesc d;
  cudaArray_t a = nullptr;
  cudaMipmappedArray_t ma = nullptr;
  cudaBindTexture(&s, &r, v, &d);
  cudaBindTexture2D(&s, &r, v, &d, s, s, s);
  cudaBindTextureToArray(&r, a, &d);
  cudaBindTextureToMipmappedArray(&r, ma, &d);
  cudaUnbindTexture(&r);
}

void Driver_TextureReferenceManagement() {
  CUaddress_mode am;
  CUtexref r;
  int i = 1;
  CUfilter_mode fm;
  unsigned int u;
  size_t s = 1;
  CUdeviceptr d;
  CUDA_ARRAY_DESCRIPTOR D;
  CUarray a;
  CUarray_format f;
  cuTexRefGetAddressMode(&am, r, i);
  cuTexRefGetFilterMode(&fm, r);
  cuTexRefGetFlags(&u, r);
  cuTexRefSetAddress(&s, r, d, s);
  cuTexRefSetAddress2D(r, &D, d, s);
  cuTexRefSetAddressMode(r, i, am);
  cuTexRefSetArray(r, a, u);
  cuTexRefSetFilterMode(r, fm);
  cuTexRefSetFlags(r, u);
  cuTexRefSetFormat(r, f, i);
}

int main() {
  Runtime_MemoryManagement();
  Runtime_TextureReferenceManagement();
  Driver_TextureReferenceManagement();
  return 0;
}
