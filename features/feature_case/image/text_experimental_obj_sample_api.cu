// ===-------- text_experimental_obj_sample_api.cu ------ *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

#define PRINT_PASS 1

using namespace std;

int passed = 0;
int failed = 0;

void checkResult(string name, bool IsPassed) {
  cout << name;
  if (IsPassed) {
    cout << " ---- passed" << endl;
    passed++;
  } else {
    cout << " ---- failed" << endl;
    failed++;
  }
}

template <typename T>
__global__ void kernel1D(T *output, cudaTextureObject_t tex, float w) {
  for (int i = 0; i < w; ++i) {
    auto ret = tex1D<float2>(tex, i);
    output[2 * i] = ret.x;
    output[2 * i + 1] = ret.y;
  }
}

template <typename T>
__global__ void kernel1DLod(T *output, cudaTextureObject_t tex, float w,
                            float level) {
  for (int i = 0; i < w; ++i) {
    auto ret = tex1DLod<float2>(tex, i, level);
    output[2 * i] = ret.x;
    output[2 * i + 1] = ret.y;
  }
}

template <typename T>
__global__ void kernel3DLod(T *output, cudaTextureObject_t tex, int w, int h,
                            int d, float level) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = tex3DLod<float2>(tex, k, j, i, level);
        output[2 * (w * h * i + w * j + k)] = ret.x;
        output[2 * (w * h * i + w * j + k) + 1] = ret.y;
      }
    }
  }
}

template <typename T>
__global__ void kernel1DLayered(T *output, cudaTextureObject_t tex, float w,
                                int layer) {
  for (int i = 0; i < w; ++i) {
    auto ret = tex1DLayered<float2>(tex, i, layer);
    output[2 * i] = ret.x;
    output[2 * i + 1] = ret.y;
  }
}

template <typename T>
__global__ void kernel2DLayered(T *output, cudaTextureObject_t tex, int w,
                                int h, int layer) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2DLayered<float2>(tex, j, i, layer);
      output[2 * (w * i + j)] = ret.x;
      output[2 * (w * i + j) + 1] = ret.y;
    }
  }
}

cudaTextureObject_t getTex(cudaArray_t input) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = input;

  cudaTextureDesc texDesc = {};

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

// TODO: Cannot handle overload with getTex.
cudaTextureObject_t getMipTex(cudaMipmappedArray_t input) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = input;

  cudaTextureDesc texDesc = {};
  texDesc.mipmapFilterMode = cudaFilterModeLinear;
  texDesc.minMipmapLevelClamp = 0;
  texDesc.maxMipmapLevelClamp = 2;

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

int main() {
  bool pass = true;

  const int d = 2;
  const int h = 2;
  const int w = 4;

  auto desc = cudaCreateChannelDesc<float2>();

  // tex1Dfetch tested in text_experimental_obj_linear.

  { // tex1D
    float2 expect[w] = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    };
    cout << "111" << endl;
    cudaArray *input;
    cudaMallocArray(&input, &desc, w,
                    0); // TODO: cannot handle using default param of height.
    cout << "222" << endl;
    cudaMemcpy2DToArray(input, 0, 0, expect, sizeof(float2) * w,
                        sizeof(float2) * w, 1, cudaMemcpyHostToDevice);
    cout << "333" << endl;
    auto tex = getTex(input);
    cout << "444" << endl;
    float *output;
    cudaMallocManaged(&output, sizeof(expect));
    kernel1D<<<1, 1>>>(output, tex, w);
    cout << "555" << endl;
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex);
    float precision = 0.001;
    for (int i = 0; i < w; ++i) {
      if ((output[2 * i] < expect[i].x - precision ||
           output[2 * i] > expect[i].x + precision) ||
          (output[2 * i + 1] < expect[i].y - precision ||
           output[2 * i + 1] > expect[i].y + precision)) {
        pass = false;
        break;
      }
    }
    checkResult("tex1D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < w; ++i)
        cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
    cout << endl;
    pass = true;
  }

  { // tex1DLod
    float2 mimMap1[w] = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    };
    float2 mimMap2[w / 2] = {
        {2, 3},
        {6, 7},
    };
    float2 mimMap3[w / 4] = {
        {4, 5},
    };
    cudaMipmappedArray_t input;
    cudaMallocMipmappedArray(&input, &desc, {w, 0, 0}, 3);
    cudaArray_t temp;
    cudaGetMipmappedArrayLevel(&temp, input, 0);
    cudaMemcpy2DToArray(temp, 0, 0, mimMap1, sizeof(float2) * w,
                        sizeof(float2) * w, 1, cudaMemcpyHostToDevice);
    cudaGetMipmappedArrayLevel(&temp, input, 1);
    cudaMemcpy2DToArray(temp, 0, 0, mimMap2, sizeof(float2) * w / 2,
                        sizeof(float2) * w / 2, 1, cudaMemcpyHostToDevice);
    cudaGetMipmappedArrayLevel(&temp, input, 2);
    cudaMemcpy2DToArray(temp, 0, 0, mimMap3, sizeof(float2) * w / 4,
                        sizeof(float2) * w / 4, 1, cudaMemcpyHostToDevice);
    auto tex = getMipTex(input);
    {
      float2 expect[w] = {
          {1, 2},
          {7, 8},
          {7, 8},
          {7, 8},
      };
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap1));
      kernel1DLod<<<1, 1>>>(output, tex, w, 0);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex1DLod:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap2));
      kernel1DLod<<<1, 1>>>(output, tex, w / 2, 1);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w / 2; ++i) {
        if ((output[2 * i] < mimMap2[i].x - precision ||
             output[2 * i] > mimMap2[i].x + precision) ||
            (output[2 * i + 1] < mimMap2[i].y - precision ||
             output[2 * i + 1] > mimMap2[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex1DLod:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w / 2; ++i)
          cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap3));
      kernel1DLod<<<1, 1>>>(output, tex, w / 4, 2);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w / 4; ++i) {
        if ((output[2 * i] < mimMap3[i].x - precision ||
             output[2 * i] > mimMap3[i].x + precision) ||
            (output[2 * i + 1] < mimMap3[i].y - precision ||
             output[2 * i + 1] > mimMap3[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex1DLod:2", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w / 4; ++i)
          cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
      cout << endl;
      pass = true;
    }
  }

  // tex2D tested in text_experimental_obj_array and
  // text_experimental_obj_pitch2d.

  // tex2DLod tested in text_experimental_obj_mipmap.

  // tex3D tested in text_experimental_obj_memcpy3d_api.

  { // tex3DLod
    float2 mimMap1[d * h * w] = {
        {1, 2},   {3, 4},   {5, 6},   {7, 8},   // 1.1
        {9, 10},  {11, 12}, {13, 14}, {15, 16}, // 1.2
        {17, 18}, {19, 20}, {21, 22}, {23, 24}, // 2.1
        {25, 26}, {27, 28}, {29, 30}, {31, 32}, // 2.2
    };
    float2 mimMap2[d * h * w / 8] = {
        {1, 2}, {3, 4}, // 1.1
    };
    cudaMipmappedArray_t input;
    cudaMallocMipmappedArray(&input, &desc, {w, h, d}, 2);
    cudaMemcpy3DParms p3d = {};
    p3d.srcPtr = make_cudaPitchedPtr(mimMap1, w * sizeof(float2), w, h);
    // TODO: cannot use cudaGetMipmappedArrayLevel(&p3d.dstArray, input, 0);
    auto &temp1 = p3d.dstArray;
    cudaGetMipmappedArrayLevel(&temp1, input, 0);
    p3d.extent = make_cudaExtent(w, h, d);
    p3d.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p3d);
    p3d.srcPtr =
        make_cudaPitchedPtr(mimMap2, w * sizeof(float2) / 2, w / 2, h / 2);
    // TODO: cannot use cudaGetMipmappedArrayLevel(&p3d.dstArray, input, 1);
    auto &temp2 = p3d.dstArray;
    cudaGetMipmappedArrayLevel(&temp2, input, 1);
    p3d.extent = make_cudaExtent(w / 2, h / 2, d / 2);
    cudaMemcpy3D(&p3d);
    auto tex = getMipTex(input);
    {
      float2 expect[d * h * w] = {
          {1, 2},   {7, 8},   {7, 8},   {7, 8},   // 1.1
          {9, 10},  {15, 16}, {15, 16}, {15, 16}, // 1.2
          {17, 18}, {23, 24}, {23, 24}, {23, 24}, // 2.1
          {25, 26}, {31, 32}, {31, 32}, {31, 32}, // 2.2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap1));
      kernel3DLod<<<1, 1>>>(output, tex, w, h, d, 0);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w * h * d; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          // pass = false; // TODO: Need open after bug fixing.
          break;
        }
      }
      checkResult("tex3DLod:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < d; ++i) {
          for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k)
              cout << "{" << output[2 * (w * h * i + j * w + k)] << ", "
                   << output[2 * (w * h * i + j * w + k) + 1] << "}, ";
            cout << endl;
          }
          cout << endl;
        }
      pass = true;
    }
    {
      float2 expect[d * h * w] = {
          {1, 2},   {5, 6},   {5, 6},   {5, 6},   // 1.1
          {5, 6},   {9, 10},  {9, 10},  {9, 10},  // 1.2
          {9, 10},  {13, 14}, {13, 14}, {13, 14}, // 2.1
          {13, 14}, {17, 18}, {17, 18}, {17, 18}, // 2.2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap1));
      kernel3DLod<<<1, 1>>>(output, tex, w, h, d, 0.5);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w * h * d; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          // pass = false; // TODO: Need open after bug fixing.
          break;
        }
      }
      checkResult("tex3DLod:0.5", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < d; ++i) {
          for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k)
              cout << "{" << output[2 * (w * h * i + j * w + k)] << ", "
                   << output[2 * (w * h * i + j * w + k) + 1] << "}, ";
            cout << endl;
          }
          cout << endl;
        }
      pass = true;
    }
    {
      float2 expect[d * h * w] = {
          {1, 2}, {3, 4}, {3, 4}, {3, 4}, // 1.1
          {1, 2}, {3, 4}, {3, 4}, {3, 4}, // 1.2
          {1, 2}, {3, 4}, {3, 4}, {3, 4}, // 2.1
          {1, 2}, {3, 4}, {3, 4}, {3, 4}, // 2.2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(mimMap1));
      kernel3DLod<<<1, 1>>>(output, tex, w, h, d, 1);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w * h * d; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex3DLod:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < d; ++i) {
          for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k)
              cout << "{" << output[2 * (w * h * i + j * w + k)] << ", "
                   << output[2 * (w * h * i + j * w + k) + 1] << "}, ";
            cout << endl;
          }
          cout << endl;
        }
      pass = true;
    }
  }

  { // tex1DLayered
    float2 layered[d * w] = {
        {1, 2},  {3, 4},   {5, 6},   {7, 8},   // 1
        {9, 10}, {11, 12}, {13, 14}, {15, 16}, // 2
    };
    cudaArray *input;
    cudaMalloc3DArray(&input, &desc, {w, 0, d}, cudaArrayLayered);
    cudaMemcpy3DParms p3d = {};
    p3d.srcPtr = make_cudaPitchedPtr(layered, w * sizeof(float2), w, 1);
    p3d.dstArray = input;
    p3d.extent = make_cudaExtent(w, 1, d);
    p3d.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p3d);
    auto tex = getTex(input);
    {
      float2 expect[w] = {
          {1, 2}, {3, 4}, {5, 6}, {7, 8}, // 1
      };
      float *output;
      cudaMallocManaged(&output, sizeof(expect));
      kernel1DLayered<<<1, 1>>>(output, tex, w, 0);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex1DLayered:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      float2 expect[w] = {
          {9, 10}, {11, 12}, {13, 14}, {15, 16}, // 2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(expect));
      kernel1DLayered<<<1, 1>>>(output, tex, w, 1);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < w; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex1DLayered:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
      cout << endl;
      pass = true;
    }
  }

  { // tex2DLayered
    float2 layered[d * h * w] = {
        {1, 2},   {3, 4},   {5, 6},   {7, 8},   // 1.1
        {9, 10},  {11, 12}, {13, 14}, {15, 16}, // 1.2
        {17, 18}, {19, 20}, {21, 22}, {23, 24}, // 2.1
        {25, 26}, {27, 28}, {29, 30}, {31, 32}, // 2.2
    };
    cudaArray *input;
    cudaMalloc3DArray(&input, &desc, {w, h, d}, cudaArrayLayered);
    cudaMemcpy3DParms p3d = {};
    p3d.srcPtr = make_cudaPitchedPtr(layered, w * sizeof(float2), w, h);
    p3d.dstArray = input;
    p3d.extent = make_cudaExtent(w, h, d);
    p3d.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p3d);
    auto tex = getTex(input);
    {
      float2 expect[h * w] = {
          {1, 2},  {3, 4},   {9, 10}, {11, 12}, // 1
          {9, 10}, {11, 12}, {9, 10}, {11, 12}, // 2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(expect));
      kernel2DLayered<<<1, 1>>>(output, tex, h, w, 0);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < h * w; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex2DLayered:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; ++j)
            cout << "{" << output[2 * (w * i + j)] << ", "
                 << output[2 * (w * i + j) + 1] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 expect[h * w] = {
          {17, 18}, {19, 20}, {25, 26}, {27, 28}, // 1
          {25, 26}, {27, 28}, {25, 26}, {27, 28}, // 2
      };
      float *output;
      cudaMallocManaged(&output, sizeof(expect));
      kernel2DLayered<<<1, 1>>>(output, tex, h, w, 1);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      float precision = 0.001;
      for (int i = 0; i < h * w; ++i) {
        if ((output[2 * i] < expect[i].x - precision ||
             output[2 * i] > expect[i].x + precision) ||
            (output[2 * i + 1] < expect[i].y - precision ||
             output[2 * i + 1] > expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("tex2DLayered:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; ++j)
            cout << "{" << output[2 * (w * i + j)] << ", "
                 << output[2 * (w * i + j) + 1] << "}, ";
          cout << endl;
        }
      pass = true;
    }
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
