// ===-------- text_experimental_obj_mipmap.cu ------- *- CUDA -* ---------===//
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

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaTextureObject_t tex, int w, int h,
                        float level) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2DLod<T>(tex, j, i, level);
      output[8 * (w * i + j)] = ret.x;
      output[8 * (w * i + j) + 1] = ret.y;
      output[8 * (w * i + j) + 2] = ret.z;
      output[8 * (w * i + j) + 3] = ret.w;
      auto ret1 = tex2DLod<T>(tex, j + 0.3, i + 0.3, level);
      output[8 * (w * i + j) + 5] = ret.x;
      output[8 * (w * i + j) + 6] = ret.y;
      output[8 * (w * i + j) + 7] = ret.z;
      output[8 * (w * i + j) + 8] = ret.w;
    }
  }
}

template <typename T, typename ArrT1, typename ArrT2>
cudaMipmappedArray_t getInput(ArrT1 &mipmap1, ArrT2 &mipmap2, size_t w,
                              size_t h, const cudaChannelFormatDesc &desc) {
  cudaMipmappedArray_t input;
  cudaMallocMipmappedArray(&input, &desc, {w, h, 0}, 2);
  cudaArray_t temp;
  cudaGetMipmappedArrayLevel(&temp, input, 0);
  cudaMemcpy2DToArray(temp, 0, 0, mipmap1, sizeof(T) * w, sizeof(T) * w, h,
                      cudaMemcpyHostToDevice);
  cudaArray_t temp1;
  cudaGetMipmappedArrayLevel(&temp1, input, 1);
  cudaMemcpy2DToArray(temp1, 0, 0, mipmap2, sizeof(T) * w / 2,
                      sizeof(T) * w / 2, h / 2, cudaMemcpyHostToDevice);
  return input;
}

cudaTextureObject_t
getTex(cudaMipmappedArray_t input, float minMipmapLevelClamp,
       float maxMipmapLevelClamp, float maxAnisotropy = 0,
       cudaTextureFilterMode mipmapFilterMode = cudaFilterModePoint) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = input;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.filterMode = cudaFilterModePoint; // TODO: Need remove this line.
  texDesc.maxAnisotropy = maxAnisotropy;
  texDesc.mipmapFilterMode = mipmapFilterMode;
  texDesc.minMipmapLevelClamp = minMipmapLevelClamp;
  texDesc.maxMipmapLevelClamp = maxMipmapLevelClamp;

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

int main() {
  bool pass = true;

  {
    const int short4H = 2;
    const int short4W = 4;
    short4 short4MimMap1[short4H * short4W] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 1
        {17, 18, 19, 20}, {21, 22, 23, 24},
        {25, 26, 27, 28}, {29, 30, 31, 32}, // 2
    };
    short4 short4MimMap2[short4H * short4W / 4] = {
        {11, 22, 33, 44}, {55, 66, 77, 88}, // 1
    };
    auto short4Input =
        getInput<short4>(short4MimMap1, short4MimMap2, short4W, short4H,
                         cudaCreateChannelDesc<short4>());
    auto short4Tex = getTex(short4Input, 0.1, 0.9);

    {
      short4 short4Expect0[short4H * short4W * 2] = {
          {1, 2, 3, 4},     {0, 1, 2, 3},     {13, 14, 15, 16},
          {0, 13, 14, 15},  {13, 14, 15, 16}, {0, 13, 14, 15},
          {13, 14, 15, 16}, {0, 13, 14, 15}, // 1
          {17, 18, 19, 20}, {0, 17, 18, 19},  {29, 30, 31, 32},
          {0, 29, 30, 31},  {29, 30, 31, 32}, {0, 29, 30, 31},
          {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
      };
      short *short4Output0;
      cudaMallocManaged(&short4Output0, sizeof(short4Expect0));
      kernel4<short4><<<1, 1>>>(short4Output0, short4Tex, short4W, short4H, 0);
      cudaDeviceSynchronize();
      for (int i = 0; i < short4W * short4H * 2; ++i) {
        if (short4Output0[4 * i] != short4Expect0[i].x ||
            short4Output0[4 * i + 1] != short4Expect0[i].y ||
            short4Output0[4 * i + 2] != short4Expect0[i].z ||
            short4Output0[4 * i + 3] != short4Expect0[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("short4:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < short4H; ++i) {
          for (int j = 0; j < short4W; ++j)
            cout << "{" << short4Output0[8 * (short4W * i + j)] << ", "
                 << short4Output0[8 * (short4W * i + j) + 1] << ", "
                 << short4Output0[8 * (short4W * i + j) + 2] << ", "
                 << short4Output0[8 * (short4W * i + j) + 3] << "}, {"
                 << short4Output0[8 * (short4W * i + j) + 4] << ", "
                 << short4Output0[8 * (short4W * i + j) + 5] << ", "
                 << short4Output0[8 * (short4W * i + j) + 6] << ", "
                 << short4Output0[8 * (short4W * i + j) + 7] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      short4 short4Expect0_3[short4H * short4W * 2] = {
          {1, 2, 3, 4},     {0, 1, 2, 3},     {13, 14, 15, 16},
          {0, 13, 14, 15},  {13, 14, 15, 16}, {0, 13, 14, 15},
          {13, 14, 15, 16}, {0, 13, 14, 15}, // 1
          {17, 18, 19, 20}, {0, 17, 18, 19},  {29, 30, 31, 32},
          {0, 29, 30, 31},  {29, 30, 31, 32}, {0, 29, 30, 31},
          {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
      };
      short *short4Output0_3;
      cudaMallocManaged(&short4Output0_3, sizeof(short4Expect0_3));
      kernel4<short4>
          <<<1, 1>>>(short4Output0_3, short4Tex, short4W, short4H, 0.3);
      cudaDeviceSynchronize();
      for (int i = 0; i < short4W * short4H * 2; ++i) {
        if (short4Output0_3[4 * i] != short4Expect0_3[i].x ||
            short4Output0_3[4 * i + 1] != short4Expect0_3[i].y ||
            short4Output0_3[4 * i + 2] != short4Expect0_3[i].z ||
            short4Output0_3[4 * i + 3] != short4Expect0_3[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("short4:0.3", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < short4H; ++i) {
          for (int j = 0; j < short4W; ++j)
            cout << "{" << short4Output0_3[8 * (short4W * i + j)] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 1] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 2] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 3] << "}, {"
                 << short4Output0_3[8 * (short4W * i + j) + 4] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 5] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 6] << ", "
                 << short4Output0_3[8 * (short4W * i + j) + 7] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      short4 short4Expect1[short4H * short4W * 2] = {
          {11, 22, 33, 44}, {0, 11, 22, 33},  {55, 66, 77, 88},
          {0, 55, 66, 77},  {55, 66, 77, 88}, {0, 55, 66, 77},
          {55, 66, 77, 88}, {0, 55, 66, 77}, // 1
          {11, 22, 33, 44}, {0, 11, 22, 33},  {55, 66, 77, 88},
          {0, 55, 66, 77},  {55, 66, 77, 88}, {0, 55, 66, 77},
          {55, 66, 77, 88}, {0, 55, 66, 77}, // 2
      };
      short *short4Output1;
      cudaMallocManaged(&short4Output1, sizeof(short4Expect1));
      kernel4<short4><<<1, 1>>>(short4Output1, short4Tex, short4W, short4H, 1);
      cudaDeviceSynchronize();
      for (int i = 0; i < short4W * short4H * 2; ++i) {
        if (short4Output1[4 * i] != short4Expect1[i].x ||
            short4Output1[4 * i + 1] != short4Expect1[i].y ||
            short4Output1[4 * i + 2] != short4Expect1[i].z ||
            short4Output1[4 * i + 3] != short4Expect1[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("short4:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < short4H; ++i) {
          for (int j = 0; j < short4W; ++j)
            cout << "{" << short4Output1[8 * (short4W * i + j)] << ", "
                 << short4Output1[8 * (short4W * i + j) + 1] << ", "
                 << short4Output1[8 * (short4W * i + j) + 2] << ", "
                 << short4Output1[8 * (short4W * i + j) + 3] << "}, {"
                 << short4Output1[8 * (short4W * i + j) + 4] << ", "
                 << short4Output1[8 * (short4W * i + j) + 5] << ", "
                 << short4Output1[8 * (short4W * i + j) + 6] << ", "
                 << short4Output1[8 * (short4W * i + j) + 7] << "}, ";
          cout << endl;
        }
      pass = true;
    }

    cudaDestroyTextureObject(short4Tex);
    cudaFreeMipmappedArray(short4Input);
  }

  {
    const int float4H = 4;
    const int float4W = 2;
    float4 float4MimMap1[float4H * float4W] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},     // 1
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 2
        {17, 18, 19, 20}, {21, 22, 23, 24}, // 3
        {25, 26, 27, 28}, {29, 30, 31, 32}, // 4
    };
    float4 float4MimMap2[float4H * float4W / 4] = {
        {11, 22, 33, 44}, // 1
        {55, 66, 77, 88}, // 2
    };
    auto *float4Input =
        getInput<float4>(float4MimMap1, float4MimMap2, float4W, float4H,
                         cudaCreateChannelDesc<float4>());

    {
      auto float4Tex = getTex(float4Input, 0.1, 0.9);

      {
        float4 float4Expect0[float4H * float4W * 2] = {
            {1, 2, 3, 4},     {0, 1, 2, 3},
            {5, 6, 7, 8},     {0, 5, 6, 7}, // 1
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 3
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0[i].x - precision ||
               float4Output0[4 * i] > float4Expect0[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Point:0", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect0_3[float4H * float4W * 2] = {
            {1, 2, 3, 4},     {0, 1, 2, 3},
            {5, 6, 7, 8},     {0, 5, 6, 7}, // 1
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 3
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0_3));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0.3);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0_3[i].x - precision ||
               float4Output0[4 * i] > float4Expect0_3[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0_3[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0_3[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0_3[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0_3[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0_3[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0_3[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Point:0.3", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect1[float4H * float4W * 2] = {
            {11, 22, 33, 44}, {0, 11, 22, 33},
            {11, 22, 33, 44}, {0, 11, 22, 33}, // 1
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 2
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 3
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect1));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 1);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect1[i].x - precision ||
               float4Output0[4 * i] > float4Expect1[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect1[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect1[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect1[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect1[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect1[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect1[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Point:1", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }

      cudaDestroyTextureObject(float4Tex);
    }
    {
      auto float4Tex = getTex(float4Input, 0.1, 0.9, 0, cudaFilterModeLinear);

      {
        float4 float4Expect0[float4H * float4W * 2] = {
            {1.97656, 3.95312, 5.92969, 7.90625},
            {0, 1.97656, 3.95312, 5.92969},
            {5.58594, 7.5625, 9.53906, 11.5156},
            {0, 5.58594, 7.5625, 9.53906}, // 1
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 2
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 3
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0[i].x - precision ||
               float4Output0[4 * i] > float4Expect0[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Linear:0", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect0_3[float4H * float4W * 2] = {
            {3.96875, 7.9375, 11.9062, 15.875},
            {0, 3.96875, 7.9375, 11.9062},
            {6.78125, 10.75, 14.7188, 18.6875},
            {0, 6.78125, 10.75, 14.7188}, // 1
            {33.9062, 37.875, 41.8438, 45.8125},
            {0, 33.9062, 37.875, 41.8438},
            {36.7188, 40.6875, 44.6562, 48.625},
            {0, 36.7188, 40.6875, 44.6562}, // 2
            {33.9062, 37.875, 41.8438, 45.8125},
            {0, 33.9062, 37.875, 41.8438},
            {36.7188, 40.6875, 44.6562, 48.625},
            {0, 36.7188, 40.6875, 44.6562}, // 3
            {33.9062, 37.875, 41.8438, 45.8125},
            {0, 33.9062, 37.875, 41.8438},
            {36.7188, 40.6875, 44.6562, 48.625},
            {0, 36.7188, 40.6875, 44.6562}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0_3));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0.3);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0_3[i].x - precision ||
               float4Output0[4 * i] > float4Expect0_3[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0_3[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0_3[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0_3[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0_3[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0_3[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0_3[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Linear:0.3", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect1[float4H * float4W * 2] = {
            {9.98438, 19.9688, 29.9531, 39.9375},
            {0, 9.98438, 19.9688, 29.9531},
            {10.3906, 20.375, 30.3594, 40.3438},
            {0, 10.3906, 20.375, 30.3594}, // 1
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 2
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 3
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect1));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 1);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect1[i].x - precision ||
               float4Output0[4 * i] > float4Expect1[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect1[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect1[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect1[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect1[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect1[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect1[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|0|Linear:1", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }

      cudaDestroyTextureObject(float4Tex);
    }
    {
      auto float4Tex = getTex(float4Input, 0.1, 0.9, 6, cudaFilterModeLinear);

      {
        float4 float4Expect0[float4H * float4W * 2] = {
            {1.97656, 3.95312, 5.92969, 7.90625},
            {0, 1.97656, 3.95312, 5.92969},
            {5.58594, 7.5625, 9.53906, 11.5156},
            {0, 5.58594, 7.5625, 9.53906}, // 1
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 2
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 3
            {27.9297, 29.9062, 31.8828, 33.8594},
            {0, 27.9297, 29.9062, 31.8828},
            {31.5391, 33.5156, 35.4922, 37.4688},
            {0, 31.5391, 33.5156, 35.4922}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0[i].x - precision ||
               float4Output0[4 * i] > float4Expect0[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|6|Linear:0", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect0_3[float4H * float4W * 2] = {
            {2.48438, 4.96875, 7.45312, 9.9375},
            {0, 2.48438, 4.96875, 7.45312},
            {5.89062, 8.375, 10.8594, 13.3438},
            {0, 5.89062, 8.375, 10.8594}, // 1
            {29.4531, 31.9375, 34.4219, 36.9062},
            {0, 29.4531, 31.9375, 34.4219},
            {32.8594, 35.3438, 37.8281, 40.3125},
            {0, 32.8594, 35.3438, 37.8281}, // 2
            {29.4531, 31.9375, 34.4219, 36.9062},
            {0, 29.4531, 31.9375, 34.4219},
            {32.8594, 35.3438, 37.8281, 40.3125},
            {0, 32.8594, 35.3438, 37.8281}, // 3
            {29.4531, 31.9375, 34.4219, 36.9062},
            {0, 29.4531, 31.9375, 34.4219},
            {32.8594, 35.3438, 37.8281, 40.3125},
            {0, 32.8594, 35.3438, 37.8281}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0_3));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0.3);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0_3[i].x - precision ||
               float4Output0[4 * i] > float4Expect0_3[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0_3[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0_3[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0_3[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0_3[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0_3[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0_3[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|6|Linear:0.3", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect1[float4H * float4W * 2] = {
            {9.98438, 19.9688, 29.9531, 39.9375},
            {0, 9.98438, 19.9688, 29.9531},
            {10.3906, 20.375, 30.3594, 40.3438},
            {0, 10.3906, 20.375, 30.3594}, // 1
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 2
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 3
            {51.9531, 61.9375, 71.9219, 81.9062},
            {0, 51.9531, 61.9375, 71.9219},
            {52.3594, 62.3438, 72.3281, 82.3125},
            {0, 52.3594, 62.3438, 72.3281}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect1));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 1);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect1[i].x - precision ||
               float4Output0[4 * i] > float4Expect1[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect1[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect1[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect1[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect1[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect1[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect1[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.9|6|Linear:1", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }

      cudaDestroyTextureObject(float4Tex);
    }
    {
      auto float4Tex = getTex(float4Input, 0.5, 0.9);

      {
        float4 float4Expect0[float4H * float4W * 2] = {
            {11, 22, 33, 44}, {0, 11, 22, 33},
            {11, 22, 33, 44}, {0, 11, 22, 33}, // 1
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 2
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 3
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0[i].x - precision ||
               float4Output0[4 * i] > float4Expect0[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.5|0.9|0|Point:0", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect0_3[float4H * float4W * 2] = {
            {11, 22, 33, 44}, {0, 11, 22, 33},
            {11, 22, 33, 44}, {0, 11, 22, 33}, // 1
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 2
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 3
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0_3));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0.3);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0_3[i].x - precision ||
               float4Output0[4 * i] > float4Expect0_3[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0_3[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0_3[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0_3[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0_3[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0_3[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0_3[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.5|0.9|0|Point:0.3", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect1[float4H * float4W * 2] = {
            {11, 22, 33, 44}, {0, 11, 22, 33},
            {11, 22, 33, 44}, {0, 11, 22, 33}, // 1
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 2
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 3
            {55, 66, 77, 88}, {0, 55, 66, 77},
            {55, 66, 77, 88}, {0, 55, 66, 77}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect1));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 1);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect1[i].x - precision ||
               float4Output0[4 * i] > float4Expect1[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect1[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect1[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect1[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect1[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect1[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect1[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.5|0.9|0|Point:1", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }

      cudaDestroyTextureObject(float4Tex);
    }
    {
      auto float4Tex = getTex(float4Input, 0.1, 0.2);

      {
        float4 float4Expect0[float4H * float4W * 2] = {
            {1, 2, 3, 4},     {0, 1, 2, 3},
            {5, 6, 7, 8},     {0, 5, 6, 7}, // 1
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 3
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0[i].x - precision ||
               float4Output0[4 * i] > float4Expect0[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.2|0|Point:0", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect0_3[float4H * float4W * 2] = {
            {1, 2, 3, 4},     {0, 1, 2, 3},
            {5, 6, 7, 8},     {0, 5, 6, 7}, // 1
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 3
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect0_3));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 0.3);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect0_3[i].x - precision ||
               float4Output0[4 * i] > float4Expect0_3[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect0_3[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect0_3[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect0_3[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect0_3[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect0_3[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect0_3[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.2|0|Point:0.3", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }
      {
        float4 float4Expect1[float4H * float4W * 2] = {
            {1, 2, 3, 4},     {0, 1, 2, 3},
            {5, 6, 7, 8},     {0, 5, 6, 7}, // 1
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 2
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 3
            {25, 26, 27, 28}, {0, 25, 26, 27},
            {29, 30, 31, 32}, {0, 29, 30, 31}, // 4
        };
        float *float4Output0;
        cudaMallocManaged(&float4Output0, sizeof(float4Expect1));
        kernel4<float4>
            <<<1, 1>>>(float4Output0, float4Tex, float4W, float4H, 1);
        cudaDeviceSynchronize();
        float precision = 0.0001;
        for (int i = 0; i < float4H * float4W * 2; ++i) {
          if ((float4Output0[4 * i] < float4Expect1[i].x - precision ||
               float4Output0[4 * i] > float4Expect1[i].x + precision) ||
              (float4Output0[4 * i + 1] < float4Expect1[i].y - precision ||
               float4Output0[4 * i + 1] > float4Expect1[i].y + precision) ||
              (float4Output0[4 * i + 2] < float4Expect1[i].z - precision ||
               float4Output0[4 * i + 2] > float4Expect1[i].z + precision) ||
              (float4Output0[4 * i + 3] < float4Expect1[i].w - precision ||
               float4Output0[4 * i + 3] > float4Expect1[i].w + precision)) {
            pass = false;
            break;
          }
        }
        checkResult("float4|0.1|0.2|0|Point:1", pass);
        if (PRINT_PASS || !pass)
          for (int i = 0; i < float4H; ++i) {
            for (int j = 0; j < float4W; ++j)
              cout << "{" << float4Output0[8 * (float4W * i + j)] << ", "
                   << float4Output0[8 * (float4W * i + j) + 1] << ", "
                   << float4Output0[8 * (float4W * i + j) + 2] << ", "
                   << float4Output0[8 * (float4W * i + j) + 3] << "}, {"
                   << float4Output0[8 * (float4W * i + j) + 4] << ", "
                   << float4Output0[8 * (float4W * i + j) + 5] << ", "
                   << float4Output0[8 * (float4W * i + j) + 6] << ", "
                   << float4Output0[8 * (float4W * i + j) + 7] << "}, ";
            cout << endl;
          }
        pass = true;
      }

      cudaDestroyTextureObject(float4Tex);
    }

    cudaFreeMipmappedArray(float4Input);
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
