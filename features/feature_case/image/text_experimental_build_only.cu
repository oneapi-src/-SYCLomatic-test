// ===-------- text_experimental_build_only.cu ------- *- CUDA -* ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>

__global__ void
CppLanguageExtensions_TextureFunctions(cudaTextureObject_t tex) {
  int i = 1, t = 1;
  float j = 1, k = 1, l = 1, m = 1;
  tex1Dfetch<float4>(tex, i);
  tex1D<short2>(tex, i);
  tex1DLod<ushort2>(tex, i, t);
  tex2D<int1>(tex, j, k);
  tex2DLod<uint1>(tex, j, k, l);
  tex3D<char4>(tex, j, k, l);
  tex3DLod<uchar4>(tex, j, k, l, m);
  tex1DLayered<uchar2>(tex, i, t);
  tex2DLayered<uint2>(tex, j, k, t);
}

void Runtime_MemoryManagement() {
  cudaChannelFormatDesc d;
  cudaExtent e;
  unsigned int u;
  cudaArray_t a = nullptr;
  cudaMipmappedArray_t m;
  cudaPitchedPtr p;
  size_t s = 1;
  void *v;
  cudaMemcpyKind k = cudaMemcpyDefault;
  cudaMemcpy3DParms pm;
  cudaMemcpy3DPeerParms ppm;
  int i = 1;
  cudaArrayGetInfo(&d, &e, &u, a);
  cudaFreeArray(a);
  cudaFreeMipmappedArray(m);
  cudaGetMipmappedArrayLevel(&a, m, u);
  cudaMalloc3D(&p, e);
  cudaMalloc3DArray(&a, &d, e, u);
  cudaMallocArray(&a, &d, s, s, u);
  cudaMallocMipmappedArray(&m, &d, e, u, i);
  cudaMallocPitch(&v, &s, s, s);
  cudaMemcpy2D(v, s, v, s, s, s, k);
  cudaMemcpy2DArrayToArray(a, s, s, a, s, s, s, s, k);
  cudaMemcpy2DAsync(v, s, v, s, s, s, k);
  cudaMemcpy2DFromArray(v, s, a, s, s, s, s, k);
  cudaMemcpy2DFromArrayAsync(v, s, a, s, s, s, s, k);
  cudaMemcpy2DToArray(a, s, s, v, s, s, s, k);
  cudaMemcpy2DToArrayAsync(a, s, s, v, s, s, s, k);
  cudaMemcpy3D(&pm);
  cudaMemcpy3DAsync(&pm);
  cudaMemcpy3DPeer(&ppm);
  cudaMemcpy3DPeerAsync(&ppm);
  cudaMemset2D(v, s, i, s, s);
  cudaMemset2DAsync(v, s, i, s, s);
  cudaMemset3D(p, i, e);
  cudaMemset3DAsync(p, i, e);
}

void Runtime_TextureObjectManagement() {
  int i = 1;
  cudaChannelFormatKind k = cudaChannelFormatKindSigned;
  cudaTextureObject_t o;
  cudaResourceDesc r;
  cudaTextureDesc t;
  // cudaResourceViewDesc v; // TODO: need support.
  cudaArray_t a = nullptr;
  cudaChannelFormatDesc d;
  cudaCreateChannelDesc(i, i, i, i, k);
  cudaCreateTextureObject(&o, &r, &t, nullptr /*&v*/);
  cudaDestroyTextureObject(o);
  cudaGetChannelDesc(&d, a);
  cudaGetTextureObjectResourceDesc(&r, o);
  cudaGetTextureObjectTextureDesc(&t, o);
}

void Driver_MemoryManagement() {
  CUarray a;
  CUDA_ARRAY3D_DESCRIPTOR D3;
  CUDA_ARRAY_DESCRIPTOR D;
  CUDA_MEMCPY2D C2;
  CUstream cs;
  CUDA_MEMCPY3D C3;
  CUDA_MEMCPY3D_PEER CP;
  size_t s;
  CUdeviceptr d;
  void *v;
  cuArray3DCreate(&a, &D3);
  cuArrayCreate(&a, &D);
  cuArrayDestroy(a);
  cuMemcpy2D(&C2);
  cuMemcpy2DAsync(&C2, cs);
  cuMemcpy3D(&C3);
  cuMemcpy3DAsync(&C3, cs);
  cuMemcpy3DPeer(&CP);
  cuMemcpy3DPeerAsync(&CP, cs);
  cuMemcpyAtoA(a, s, a, s, s);
  cuMemcpyAtoD(d, a, s, s);
  cuMemcpyAtoH(&v, a, s, s);
  cuMemcpyAtoHAsync(&v, a, s, s, cs);
  cuMemcpyDtoA(a, s, d, s);
  cuMemcpyDtoD(d, d, s);
  cuMemcpyDtoDAsync(d, d, s, cs);
  cuMemcpyDtoH(v, d, s);
  cuMemcpyDtoHAsync(v, d, s, cs);
  cuMemcpyHtoA(a, s, v, s);
  cuMemcpyHtoAAsync(a, s, v, s, cs);
  cuMemcpyHtoD(d, v, s);
  cuMemcpyHtoDAsync(d, v, s, cs);
}

void Driver_TextureObjectManagement() {
  CUtexObject o;
  CUDA_RESOURCE_DESC R;
  CUDA_TEXTURE_DESC T;
  // CUDA_RESOURCE_VIEW_DESC V; // TODO: need support.
  cuTexObjectCreate(&o, &R, &T, nullptr /*&V*/);
  cuTexObjectDestroy(o);
  cuTexObjectGetResourceDesc(&R, o);
  cuTexObjectGetTextureDesc(&T, o);
}

int main() {
  Runtime_MemoryManagement();
  Runtime_TextureObjectManagement();
  Driver_MemoryManagement();
  Driver_TextureObjectManagement();
  cudaTextureObject_t tex;
  CppLanguageExtensions_TextureFunctions<<<1, 1>>>(tex);
  return 0;
}
