// ===------------- text_obj_array.cu ---------- *- CUDA -* ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

using namespace std;

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaTextureObject_t tex, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2D<T>(tex, j, i);
      output[4 * (w * i + j)] = ret.x;
      output[4 * (w * i + j) + 1] = ret.y;
      output[4 * (w * i + j) + 2] = ret.z;
      output[4 * (w * i + j) + 3] = ret.w;
    }
  }
}

template <typename T, typename ArrT>
cudaArray *getInput(ArrT &expect, size_t w, size_t h,
                    const cudaChannelFormatDesc &desc) {
  cudaArray *input;
  cudaMallocArray(&input, &desc, w, h);
  cudaMemcpy2DToArray(input, 0, 0, expect, sizeof(T) * w, sizeof(T) * w, h,
                      cudaMemcpyHostToDevice);
  return input;
}

cudaTextureObject_t
getTex(cudaArray_t input,
       cudaTextureAddressMode addressMode = cudaAddressModeWrap,
       cudaTextureFilterMode textureFilterMode = cudaFilterModePoint,
       int normalizedCoords = 0) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = input;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

int main() {
  const int h = 2;
  const int w = 4;
  short4 expect[h * w] = {
      {1, 2, 3, 4},     {5, 6, 7, 8},     {9, 10, 11, 12},  {13, 14, 15, 16},
      {17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32},
  };
  auto *short4Input =
      getInput<short4>(expect, w, h, cudaCreateChannelDesc<short4>());
  short *output;
  cudaMallocManaged(&output, sizeof(expect));
  auto short4Tex = getTex(short4Input);
  kernel4<short4><<<1, 1>>>(output, short4Tex, w, h);
  cudaDeviceSynchronize();
  cudaDestroyTextureObject(short4Tex);
  cudaFreeArray(short4Input);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      cout << "{" << output[4 * (w * i + j)] << ", "
           << output[4 * (w * i + j) + 1] << ", " << output[4 * (w * i + j) + 2]
           << ", " << output[4 * (w * i + j) + 3] << "}," << endl;
    }
  }
  cout << "short4 test ";
  for (int i = 0; i < w * h; ++i) {
    if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
        output[4 * i + 2] != expect[i].z || output[4 * i + 3] != expect[i].w) {
      cout << "failed!" << endl;
      return 1;
    }
  }
  cout << "passed!" << endl;
  return 0;
}
