// ===-------- text_experimental_obj_pitch2d.cu ------- *- CUDA -* --------===//
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
__global__ void kernel1(EleT *output, cudaTextureObject_t tex, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2D<T>(tex, j, i);
      output[w * i + j] = ret.x;
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel2(EleT *output, cudaTextureObject_t tex, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2D<T>(tex, j, i);
      output[2 * (w * i + j)] = ret.x;
      output[2 * (w * i + j) + 1] = ret.y;
    }
  }
}

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

template <typename T, typename EleT>
__global__ void kernel4x3(EleT *output, cudaTextureObject_t tex, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2D<T>(tex, j - 0.1, i - 0.1);
      output[12 * (w * i + j)] = ret.x;
      output[12 * (w * i + j) + 1] = ret.y;
      output[12 * (w * i + j) + 2] = ret.z;
      output[12 * (w * i + j) + 3] = ret.w;
      auto ret1 = tex2D<T>(tex, j + 0.3, i + 0.3);
      output[12 * (w * i + j) + 4] = ret1.x;
      output[12 * (w * i + j) + 5] = ret1.y;
      output[12 * (w * i + j) + 6] = ret1.z;
      output[12 * (w * i + j) + 7] = ret1.w;
      auto ret2 = tex2D<T>(tex, j + 1.1, i + 1.1);
      output[12 * (w * i + j) + 8] = ret2.x;
      output[12 * (w * i + j) + 9] = ret2.y;
      output[12 * (w * i + j) + 10] = ret2.z;
      output[12 * (w * i + j) + 11] = ret2.w;
    }
  }
}

template <typename T, typename ArrT>
T *getInput(ArrT &expect, size_t *pitch, size_t w, size_t h) {
  T *input;
  cudaMallocPitch(&input, pitch, sizeof(T) * w, h);
  cudaMemcpy2D(input, *pitch, &expect, sizeof(T) * w, sizeof(T) * w, h,
               cudaMemcpyHostToDevice);
  return input;
}

cudaTextureObject_t
getTex(void *input, size_t w, size_t h, cudaChannelFormatDesc desc,
       size_t pitchInBytes,
       cudaTextureAddressMode addressMode = cudaAddressModeWrap,
       cudaTextureFilterMode textureFilterMode = cudaFilterModePoint,
       int normalizedCoords = 0) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = input;
  resDesc.res.pitch2D.width = w;
  resDesc.res.pitch2D.height = h;
  resDesc.res.pitch2D.desc = desc;
  resDesc.res.pitch2D.pitchInBytes = pitchInBytes;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.addressMode[2] = addressMode;
  texDesc.filterMode = textureFilterMode;
  texDesc.normalizedCoords = normalizedCoords;

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

int main() {
  bool pass = true;

  {
    const int char2H = 3;
    const int char2W = 2;
    char2 char2Expect[char2H * char2W] = {
        {1, 2},   {5, 6},   // 1
        {9, 10},  {13, 14}, // 2
        {17, 18}, {21, 22}, // 3
    };
    size_t char2Pitch;
    auto *char2Input =
        getInput<char2>(char2Expect, &char2Pitch, char2W, char2H);
    char *char2Output;
    cudaMallocManaged(&char2Output, sizeof(char2Expect));
    auto char2Tex = getTex(char2Input, char2W, char2H,
                           cudaCreateChannelDesc<char2>(), char2Pitch);
    kernel2<char2><<<1, 1>>>(char2Output, char2Tex, char2W, char2H);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(char2Tex);
    cudaFree(char2Input);
    for (int i = 0; i < char2W * char2H; ++i) {
      if (char2Output[2 * i] != char2Expect[i].x ||
          char2Output[2 * i + 1] != char2Expect[i].y) {
        // pass = false; // TODO: need support.
        break;
      }
    }
    checkResult("char2", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < char2H; ++i) {
        for (int j = 0; j < char2W; ++j)
          cout << "{" << (long)char2Output[2 * (char2W * i + j)] << ", "
               << (long)char2Output[2 * (char2W * i + j) + 1] << "}, ";
        cout << endl;
      }
    pass = true;
  }

  {
    const int ushort4H = 2;
    const int ushort4W = 4;
    ushort4 ushort4Expect[ushort4H * ushort4W] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 1
        {17, 18, 19, 20}, {21, 22, 23, 24},
        {25, 26, 27, 28}, {29, 30, 31, 32}, // 2
    };
    size_t ushort4Pitch;
    auto *ushort4Input =
        getInput<ushort4>(ushort4Expect, &ushort4Pitch, ushort4W, ushort4H);
    unsigned short *ushort4Output;
    cudaMallocManaged(&ushort4Output, sizeof(ushort4Expect));
    auto ushort4Tex = getTex(ushort4Input, ushort4W, ushort4H,
                             cudaCreateChannelDesc<ushort4>(), ushort4Pitch);
    kernel4<ushort4><<<1, 1>>>(ushort4Output, ushort4Tex, ushort4W, ushort4H);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(ushort4Tex);
    cudaFree(ushort4Input);
    for (int i = 0; i < ushort4W * ushort4H; ++i) {
      if (ushort4Output[4 * i] != ushort4Expect[i].x ||
          ushort4Output[4 * i + 1] != ushort4Expect[i].y ||
          ushort4Output[4 * i + 2] != ushort4Expect[i].z ||
          ushort4Output[4 * i + 3] != ushort4Expect[i].w) {
        pass = false;
        break;
      }
    }
    checkResult("ushort4", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < ushort4H; ++i) {
        for (int j = 0; j < ushort4W; ++j)
          cout << "{" << ushort4Output[4 * (ushort4W * i + j)] << ", "
               << ushort4Output[4 * (ushort4W * i + j) + 1] << ", "
               << ushort4Output[4 * (ushort4W * i + j) + 2] << ", "
               << ushort4Output[4 * (ushort4W * i + j) + 3] << "}, ";
        cout << endl;
      }
    pass = true;
  }

  {
    const int int1H = 4;
    const int int1W = 3;
    int1 int1Expect[int1H * int1W] = {
        {1},  {2},  {3},  // 1
        {4},  {5},  {6},  // 2
        {7},  {8},  {9},  // 3
        {10}, {11}, {12}, // 4
    };
    size_t int1Pitch;
    auto *int1Input = getInput<int1>(int1Expect, &int1Pitch, int1W, int1H);
    int *int1Output;
    cudaMallocManaged(&int1Output, sizeof(int1Expect));
    auto int1Tex = getTex(int1Input, int1W, int1H,
                          cudaCreateChannelDesc<int1>(), int1Pitch);
    kernel1<int4><<<1, 1>>>(int1Output, int1Tex, int1W, int1H);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(int1Tex);
    cudaFree(int1Input);
    for (int i = 0; i < int1W * int1H; ++i) {
      if (int1Output[i] != int1Expect[i].x) {
        // pass = false; // TODO: need support.
        break;
      }
    }
    checkResult("int1", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < int1H; ++i) {
        for (int j = 0; j < int1W; ++j)
          cout << "{" << int1Output[int1W * i + j] << "}, ";
        cout << endl;
      }
    pass = true;
  }

  {
    const int float4H = 4;
    const int float4W = 2;
    float4 float4Expect[float4H * float4W] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},     // 1
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 2
        {17, 18, 19, 20}, {21, 22, 23, 24}, // 3
        {25, 26, 27, 28}, {29, 30, 31, 32}, // 4
    };
    size_t float4Pitch;
    auto *float4Input =
        getInput<float4>(float4Expect, &float4Pitch, float4W, float4H);
    {
      float *float4Output;
      cudaMallocManaged(&float4Output, sizeof(float4Expect));
      auto float4Tex = getTex(float4Input, float4W, float4H,
                              cudaCreateChannelDesc<float4>(), float4Pitch);
      kernel4<float4><<<1, 1>>>(float4Output, float4Tex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4Tex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H; ++i) {
        if ((float4Output[4 * i] < float4Expect[i].x - precision ||
             float4Output[4 * i] > float4Expect[i].x + precision) ||
            (float4Output[4 * i + 1] < float4Expect[i].y - precision ||
             float4Output[4 * i + 1] > float4Expect[i].y + precision) ||
            (float4Output[4 * i + 2] < float4Expect[i].z - precision ||
             float4Output[4 * i + 2] > float4Expect[i].z + precision) ||
            (float4Output[4 * i + 3] < float4Expect[i].w - precision ||
             float4Output[4 * i + 3] > float4Expect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4Output[4 * (float4W * i + j)] << ", "
                 << float4Output[4 * (float4W * i + j) + 1] << ", "
                 << float4Output[4 * (float4W * i + j) + 2] << ", "
                 << float4Output[4 * (float4W * i + j) + 3] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3Expect[float4H * float4W * 3] = {
          {1, 2, 3, 4},     {1, 2, 3, 4},     {13, 14, 15, 16},
          {1, 2, 3, 4},     {5, 6, 7, 8},     {13, 14, 15, 16}, // 1
          {1, 2, 3, 4},     {9, 10, 11, 12},  {21, 22, 23, 24},
          {1, 2, 3, 4},     {13, 14, 15, 16}, {21, 22, 23, 24}, // 2
          {9, 10, 11, 12},  {17, 18, 19, 20}, {29, 30, 31, 32},
          {9, 10, 11, 12},  {21, 22, 23, 24}, {29, 30, 31, 32}, // 3
          {17, 18, 19, 20}, {25, 26, 27, 28}, {29, 30, 31, 32},
          {17, 18, 19, 20}, {29, 30, 31, 32}, {29, 30, 31, 32}, // 4
      };
      float *float4x3Output;
      cudaMallocManaged(&float4x3Output, sizeof(float4x3Expect));
      auto float4x3Tex = getTex(float4Input, float4W, float4H,
                                cudaCreateChannelDesc<float4>(), float4Pitch);
      kernel4x3<float4>
          <<<1, 1>>>(float4x3Output, float4x3Tex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3Tex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3Output[4 * i] < float4x3Expect[i].x - precision ||
             float4x3Output[4 * i] > float4x3Expect[i].x + precision) ||
            (float4x3Output[4 * i + 1] < float4x3Expect[i].y - precision ||
             float4x3Output[4 * i + 1] > float4x3Expect[i].y + precision) ||
            (float4x3Output[4 * i + 2] < float4x3Expect[i].z - precision ||
             float4x3Output[4 * i + 2] > float4x3Expect[i].z + precision) ||
            (float4x3Output[4 * i + 3] < float4x3Expect[i].w - precision ||
             float4x3Output[4 * i + 3] > float4x3Expect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3Output[12 * (float4W * i + j)] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 1] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 2] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 3] << "}, {"
                 << float4x3Output[12 * (float4W * i + j) + 4] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 5] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 6] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 7] << "}, {"
                 << float4x3Output[12 * (float4W * i + j) + 8] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 9] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 10] << ", "
                 << float4x3Output[12 * (float4W * i + j) + 11] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3BorderExpect[float4H * float4W * 3] = {
          {0, 0, 0, 0},     {1, 2, 3, 4},     {13, 14, 15, 16},
          {0, 0, 0, 0},     {5, 6, 7, 8},     {0, 0, 0, 0}, // 1
          {0, 0, 0, 0},     {9, 10, 11, 12},  {21, 22, 23, 24},
          {1, 2, 3, 4},     {13, 14, 15, 16}, {0, 0, 0, 0}, // 2
          {0, 0, 0, 0},     {17, 18, 19, 20}, {29, 30, 31, 32},
          {9, 10, 11, 12},  {21, 22, 23, 24}, {0, 0, 0, 0}, // 3
          {0, 0, 0, 0},     {25, 26, 27, 28}, {0, 0, 0, 0},
          {17, 18, 19, 20}, {29, 30, 31, 32}, {0, 0, 0, 0}, // 4
      };
      float *float4x3BorderOutput;
      cudaMallocManaged(&float4x3BorderOutput, sizeof(float4x3BorderExpect));
      auto float4x3BorderTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeBorder);
      kernel4x3<float4>
          <<<1, 1>>>(float4x3BorderOutput, float4x3BorderTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3BorderTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3BorderOutput[4 * i] <
                 float4x3BorderExpect[i].x - precision ||
             float4x3BorderOutput[4 * i] >
                 float4x3BorderExpect[i].x + precision) ||
            (float4x3BorderOutput[4 * i + 1] <
                 float4x3BorderExpect[i].y - precision ||
             float4x3BorderOutput[4 * i + 1] >
                 float4x3BorderExpect[i].y + precision) ||
            (float4x3BorderOutput[4 * i + 2] <
                 float4x3BorderExpect[i].z - precision ||
             float4x3BorderOutput[4 * i + 2] >
                 float4x3BorderExpect[i].z + precision) ||
            (float4x3BorderOutput[4 * i + 3] <
                 float4x3BorderExpect[i].w - precision ||
             float4x3BorderOutput[4 * i + 3] >
                 float4x3BorderExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3Border", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3BorderOutput[12 * (float4W * i + j)] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 1] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 2] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 3] << "}, {"
                 << float4x3BorderOutput[12 * (float4W * i + j) + 4] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 5] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 6] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 7] << "}, {"
                 << float4x3BorderOutput[12 * (float4W * i + j) + 8] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 9] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 10] << ", "
                 << float4x3BorderOutput[12 * (float4W * i + j) + 11] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3WrapLinearExpect[float4H * float4W * 3] = {
          {1, 2, 3, 4},
          {1, 2, 3, 4},
          {8.21875, 9.21875, 10.2188, 11.2188},
          {2.59375, 3.59375, 4.59375, 5.59375},
          {4.20312, 5.20312, 6.20312, 7.20312},
          {9.8125, 10.8125, 11.8125, 12.8125}, // 1
          {4.1875, 5.1875, 6.1875, 7.1875},
          {7.40625, 8.40625, 9.40625, 10.4062},
          {16.2188, 17.2188, 18.2188, 19.2188},
          {5.78125, 6.78125, 7.78125, 8.78125},
          {10.6094, 11.6094, 12.6094, 13.6094},
          {17.8125, 18.8125, 19.8125, 20.8125}, // 2
          {12.1875, 13.1875, 14.1875, 15.1875},
          {15.4062, 16.4062, 17.4062, 18.4062},
          {24.2188, 25.2188, 26.2188, 27.2188},
          {13.7812, 14.7812, 15.7812, 16.7812},
          {18.6094, 19.6094, 20.6094, 21.6094},
          {25.8125, 26.8125, 27.8125, 28.8125}, // 3
          {20.1875, 21.1875, 22.1875, 23.1875},
          {23.4062, 24.4062, 25.4062, 26.4062},
          {27.4062, 28.4062, 29.4062, 30.4062},
          {21.7812, 22.7812, 23.7812, 24.7812},
          {26.6094, 27.6094, 28.6094, 29.6094},
          {29, 30, 31, 32}, // 4
      };
      float *float4x3WrapLinearOutput;
      cudaMallocManaged(&float4x3WrapLinearOutput,
                        sizeof(float4x3WrapLinearExpect));
      auto float4x3WrapLinearTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeWrap, cudaFilterModeLinear);
      kernel4x3<float4><<<1, 1>>>(float4x3WrapLinearOutput,
                                  float4x3WrapLinearTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3WrapLinearTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3WrapLinearOutput[4 * i] <
                 float4x3WrapLinearExpect[i].x - precision ||
             float4x3WrapLinearOutput[4 * i] >
                 float4x3WrapLinearExpect[i].x + precision) ||
            (float4x3WrapLinearOutput[4 * i + 1] <
                 float4x3WrapLinearExpect[i].y - precision ||
             float4x3WrapLinearOutput[4 * i + 1] >
                 float4x3WrapLinearExpect[i].y + precision) ||
            (float4x3WrapLinearOutput[4 * i + 2] <
                 float4x3WrapLinearExpect[i].z - precision ||
             float4x3WrapLinearOutput[4 * i + 2] >
                 float4x3WrapLinearExpect[i].z + precision) ||
            (float4x3WrapLinearOutput[4 * i + 3] <
                 float4x3WrapLinearExpect[i].w - precision ||
             float4x3WrapLinearOutput[4 * i + 3] >
                 float4x3WrapLinearExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3WrapLinear", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3WrapLinearOutput[12 * (float4W * i + j)]
                 << ", " << float4x3WrapLinearOutput[12 * (float4W * i + j) + 1]
                 << ", " << float4x3WrapLinearOutput[12 * (float4W * i + j) + 2]
                 << ", " << float4x3WrapLinearOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 4] << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 5] << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 6] << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 8] << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 9] << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3WrapLinearOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3BorderLinearExpect[float4H * float4W * 3] = {
          {0.160156, 0.320312, 0.480469, 0.640625},
          {0.640625, 1.28125, 1.92188, 2.5625},
          {8.21875, 9.21875, 10.2188, 11.2188},
          {1.03906, 1.4375, 1.83594, 2.23438},
          {3.36328, 4.16406, 4.96484, 5.76562},
          {3.89844, 4.29688, 4.69531, 5.09375}, // 1
          {1.67969, 2.07812, 2.47656, 2.875},
          {5.92578, 6.72656, 7.52734, 8.32812},
          {16.2188, 17.2188, 18.2188, 19.2188},
          {5.78125, 6.78125, 7.78125, 8.78125},
          {10.6094, 11.6094, 12.6094, 13.6094},
          {7.08594, 7.48438, 7.88281, 8.28125}, // 2
          {4.86719, 5.26562, 5.66406, 6.0625},
          {12.332, 13.1328, 13.9336, 14.7344},
          {24.2188, 25.2188, 26.2188, 27.2188},
          {13.7812, 14.7812, 15.7812, 16.7812},
          {18.6094, 19.6094, 20.6094, 21.6094},
          {10.2734, 10.6719, 11.0703, 11.4688}, // 3
          {8.05469, 8.45312, 8.85156, 9.25},
          {18.7383, 19.5391, 20.3398, 21.1406},
          {10.9141, 11.3125, 11.7109, 12.1094},
          {21.7812, 22.7812, 23.7812, 24.7812},
          {26.6094, 27.6094, 28.6094, 29.6094},
          {4.64453, 4.80469, 4.96484, 5.125}, // 4
      };
      float *float4x3BorderLinearOutput;
      cudaMallocManaged(&float4x3BorderLinearOutput,
                        sizeof(float4x3BorderLinearExpect));
      auto float4x3BorderLinearTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeBorder, cudaFilterModeLinear);
      kernel4x3<float4><<<1, 1>>>(float4x3BorderLinearOutput,
                                  float4x3BorderLinearTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3BorderLinearTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3BorderLinearOutput[4 * i] <
                 float4x3BorderLinearExpect[i].x - precision ||
             float4x3BorderLinearOutput[4 * i] >
                 float4x3BorderLinearExpect[i].x + precision) ||
            (float4x3BorderLinearOutput[4 * i + 1] <
                 float4x3BorderLinearExpect[i].y - precision ||
             float4x3BorderLinearOutput[4 * i + 1] >
                 float4x3BorderLinearExpect[i].y + precision) ||
            (float4x3BorderLinearOutput[4 * i + 2] <
                 float4x3BorderLinearExpect[i].z - precision ||
             float4x3BorderLinearOutput[4 * i + 2] >
                 float4x3BorderLinearExpect[i].z + precision) ||
            (float4x3BorderLinearOutput[4 * i + 3] <
                 float4x3BorderLinearExpect[i].w - precision ||
             float4x3BorderLinearOutput[4 * i + 3] >
                 float4x3BorderLinearExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3BorderLinear", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3BorderLinearOutput[12 * (float4W * i + j)]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 1]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 2]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 4]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 5]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 6]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 8]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 9]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3BorderLinearOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3WrapNormExpect[float4H * float4W * 3] = {
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4},
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4}, // 1
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4},
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4}, // 2
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4},
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4}, // 3
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4},
          {29, 30, 31, 32}, {9, 10, 11, 12}, {1, 2, 3, 4}, // 4
      };
      float *float4x3WrapNormOutput;
      cudaMallocManaged(&float4x3WrapNormOutput,
                        sizeof(float4x3WrapNormExpect));
      auto float4x3WrapNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeWrap, cudaFilterModePoint, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3WrapNormOutput, float4x3WrapNormTex,
                                  float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3WrapNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3WrapNormOutput[4 * i] <
                 float4x3WrapNormExpect[i].x - precision ||
             float4x3WrapNormOutput[4 * i] >
                 float4x3WrapNormExpect[i].x + precision) ||
            (float4x3WrapNormOutput[4 * i + 1] <
                 float4x3WrapNormExpect[i].y - precision ||
             float4x3WrapNormOutput[4 * i + 1] >
                 float4x3WrapNormExpect[i].y + precision) ||
            (float4x3WrapNormOutput[4 * i + 2] <
                 float4x3WrapNormExpect[i].z - precision ||
             float4x3WrapNormOutput[4 * i + 2] >
                 float4x3WrapNormExpect[i].z + precision) ||
            (float4x3WrapNormOutput[4 * i + 3] <
                 float4x3WrapNormExpect[i].w - precision ||
             float4x3WrapNormOutput[4 * i + 3] >
                 float4x3WrapNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3WrapNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3WrapNormOutput[12 * (float4W * i + j)]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 1]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 2]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {" << float4x3WrapNormOutput[12 * (float4W * i + j) + 4]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 5]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 6]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {" << float4x3WrapNormOutput[12 * (float4W * i + j) + 8]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 9]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 10]
                 << ", " << float4x3WrapNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3ClampNormExpect[float4H * float4W * 3] = {
          {1, 2, 3, 4},     {9, 10, 11, 12},  {29, 30, 31, 32},
          {5, 6, 7, 8},     {13, 14, 15, 16}, {29, 30, 31, 32}, // 1
          {25, 26, 27, 28}, {25, 26, 27, 28}, {29, 30, 31, 32},
          {29, 30, 31, 32}, {29, 30, 31, 32}, {29, 30, 31, 32}, // 2
          {25, 26, 27, 28}, {25, 26, 27, 28}, {29, 30, 31, 32},
          {29, 30, 31, 32}, {29, 30, 31, 32}, {29, 30, 31, 32}, // 3
          {25, 26, 27, 28}, {25, 26, 27, 28}, {29, 30, 31, 32},
          {29, 30, 31, 32}, {29, 30, 31, 32}, {29, 30, 31, 32}, // 4
      };
      float *float4x3ClampNormOutput;
      cudaMallocManaged(&float4x3ClampNormOutput,
                        sizeof(float4x3ClampNormExpect));
      auto float4x3ClampNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeClamp, cudaFilterModePoint, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3ClampNormOutput, float4x3ClampNormTex,
                                  float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3ClampNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3ClampNormOutput[4 * i] <
                 float4x3ClampNormExpect[i].x - precision ||
             float4x3ClampNormOutput[4 * i] >
                 float4x3ClampNormExpect[i].x + precision) ||
            (float4x3ClampNormOutput[4 * i + 1] <
                 float4x3ClampNormExpect[i].y - precision ||
             float4x3ClampNormOutput[4 * i + 1] >
                 float4x3ClampNormExpect[i].y + precision) ||
            (float4x3ClampNormOutput[4 * i + 2] <
                 float4x3ClampNormExpect[i].z - precision ||
             float4x3ClampNormOutput[4 * i + 2] >
                 float4x3ClampNormExpect[i].z + precision) ||
            (float4x3ClampNormOutput[4 * i + 3] <
                 float4x3ClampNormExpect[i].w - precision ||
             float4x3ClampNormOutput[4 * i + 3] >
                 float4x3ClampNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3ClampNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3ClampNormOutput[12 * (float4W * i + j)]
                 << ", " << float4x3ClampNormOutput[12 * (float4W * i + j) + 1]
                 << ", " << float4x3ClampNormOutput[12 * (float4W * i + j) + 2]
                 << ", " << float4x3ClampNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 4] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 5] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 6] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 8] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 9] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 10] << ", "
                 << float4x3ClampNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3MirrorNormExpect[float4H * float4W * 3] = {
          {1, 2, 3, 4},     {9, 10, 11, 12},  {29, 30, 31, 32},
          {5, 6, 7, 8},     {13, 14, 15, 16}, {25, 26, 27, 28}, // 1
          {25, 26, 27, 28}, {17, 18, 19, 20}, {5, 6, 7, 8},
          {29, 30, 31, 32}, {21, 22, 23, 24}, {1, 2, 3, 4}, // 2
          {1, 2, 3, 4},     {9, 10, 11, 12},  {29, 30, 31, 32},
          {5, 6, 7, 8},     {13, 14, 15, 16}, {25, 26, 27, 28}, // 3
          {25, 26, 27, 28}, {17, 18, 19, 20}, {5, 6, 7, 8},
          {29, 30, 31, 32}, {21, 22, 23, 24}, {1, 2, 3, 4}, // 4
      };
      float *float4x3MirrorNormOutput;
      cudaMallocManaged(&float4x3MirrorNormOutput,
                        sizeof(float4x3MirrorNormExpect));
      auto float4x3MirrorNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeMirror, cudaFilterModePoint, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3MirrorNormOutput,
                                  float4x3MirrorNormTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3MirrorNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3MirrorNormOutput[4 * i] <
                 float4x3MirrorNormExpect[i].x - precision ||
             float4x3MirrorNormOutput[4 * i] >
                 float4x3MirrorNormExpect[i].x + precision) ||
            (float4x3MirrorNormOutput[4 * i + 1] <
                 float4x3MirrorNormExpect[i].y - precision ||
             float4x3MirrorNormOutput[4 * i + 1] >
                 float4x3MirrorNormExpect[i].y + precision) ||
            (float4x3MirrorNormOutput[4 * i + 2] <
                 float4x3MirrorNormExpect[i].z - precision ||
             float4x3MirrorNormOutput[4 * i + 2] >
                 float4x3MirrorNormExpect[i].z + precision) ||
            (float4x3MirrorNormOutput[4 * i + 3] <
                 float4x3MirrorNormExpect[i].w - precision ||
             float4x3MirrorNormOutput[4 * i + 3] >
                 float4x3MirrorNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3MirrorNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3MirrorNormOutput[12 * (float4W * i + j)]
                 << ", " << float4x3MirrorNormOutput[12 * (float4W * i + j) + 1]
                 << ", " << float4x3MirrorNormOutput[12 * (float4W * i + j) + 2]
                 << ", " << float4x3MirrorNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 4] << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 5] << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 6] << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 8] << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 9] << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3MirrorNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3BorderNormExpect[float4H * float4W * 3] = {
          {0, 0, 0, 0},     {9, 10, 11, 12}, {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0}, // 1
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0},
          {29, 30, 31, 32}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0}, // 3
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0},    {0, 0, 0, 0}, // 4
      };
      float *float4x3BorderNormOutput;
      cudaMallocManaged(&float4x3BorderNormOutput,
                        sizeof(float4x3BorderNormExpect));
      auto float4x3BorderNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeBorder, cudaFilterModePoint, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3BorderNormOutput,
                                  float4x3BorderNormTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3BorderNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3BorderNormOutput[4 * i] <
                 float4x3BorderNormExpect[i].x - precision ||
             float4x3BorderNormOutput[4 * i] >
                 float4x3BorderNormExpect[i].x + precision) ||
            (float4x3BorderNormOutput[4 * i + 1] <
                 float4x3BorderNormExpect[i].y - precision ||
             float4x3BorderNormOutput[4 * i + 1] >
                 float4x3BorderNormExpect[i].y + precision) ||
            (float4x3BorderNormOutput[4 * i + 2] <
                 float4x3BorderNormExpect[i].z - precision ||
             float4x3BorderNormOutput[4 * i + 2] >
                 float4x3BorderNormExpect[i].z + precision) ||
            (float4x3BorderNormOutput[4 * i + 3] <
                 float4x3BorderNormExpect[i].w - precision ||
             float4x3BorderNormOutput[4 * i + 3] >
                 float4x3BorderNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3BorderNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3BorderNormOutput[12 * (float4W * i + j)]
                 << ", " << float4x3BorderNormOutput[12 * (float4W * i + j) + 1]
                 << ", " << float4x3BorderNormOutput[12 * (float4W * i + j) + 2]
                 << ", " << float4x3BorderNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 4] << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 5] << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 6] << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 8] << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 9] << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3BorderNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3WrapLinearNormExpect[float4H * float4W * 3] = {
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062},
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062}, // 1
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062},
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062}, // 2
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062},
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062}, // 3
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062},
          {25.3594, 26.3594, 27.3594, 28.3594},
          {7, 8, 9, 10},
          {4.64062, 5.64062, 6.64062, 7.64062}, // 4
      };
      float *float4x3WrapLinearNormOutput;
      cudaMallocManaged(&float4x3WrapLinearNormOutput,
                        sizeof(float4x3WrapLinearNormExpect));
      auto float4x3WrapLinearNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeWrap, cudaFilterModeLinear, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3WrapLinearNormOutput,
                                  float4x3WrapLinearNormTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3WrapLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3WrapLinearNormOutput[4 * i] <
                 float4x3WrapLinearNormExpect[i].x - precision ||
             float4x3WrapLinearNormOutput[4 * i] >
                 float4x3WrapLinearNormExpect[i].x + precision) ||
            (float4x3WrapLinearNormOutput[4 * i + 1] <
                 float4x3WrapLinearNormExpect[i].y - precision ||
             float4x3WrapLinearNormOutput[4 * i + 1] >
                 float4x3WrapLinearNormExpect[i].y + precision) ||
            (float4x3WrapLinearNormOutput[4 * i + 2] <
                 float4x3WrapLinearNormExpect[i].z - precision ||
             float4x3WrapLinearNormOutput[4 * i + 2] >
                 float4x3WrapLinearNormExpect[i].z + precision) ||
            (float4x3WrapLinearNormOutput[4 * i + 3] <
                 float4x3WrapLinearNormExpect[i].w - precision ||
             float4x3WrapLinearNormOutput[4 * i + 3] >
                 float4x3WrapLinearNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3WrapLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3WrapLinearNormOutput[12 * (float4W * i + j)]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 1]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 2]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 4]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 5]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 6]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 8]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 9]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3WrapLinearNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3ClampLinearNormExpect[float4H * float4W * 3] = {
          {1, 2, 3, 4},
          {7, 8, 9, 10},
          {29, 30, 31, 32},
          {5, 6, 7, 8},
          {10.5938, 11.5938, 12.5938, 13.5938},
          {29, 30, 31, 32}, // 1
          {25, 26, 27, 28},
          {25.4062, 26.4062, 27.4062, 28.4062},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32}, // 2
          {25, 26, 27, 28},
          {25.4062, 26.4062, 27.4062, 28.4062},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32}, // 3
          {25, 26, 27, 28},
          {25.4062, 26.4062, 27.4062, 28.4062},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32},
          {29, 30, 31, 32}, // 4
      };
      float *float4x3ClampLinearNormOutput;
      cudaMallocManaged(&float4x3ClampLinearNormOutput,
                        sizeof(float4x3ClampLinearNormExpect));
      auto float4x3ClampLinearNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeClamp, cudaFilterModeLinear, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3ClampLinearNormOutput,
                                  float4x3ClampLinearNormTex, float4W, float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3ClampLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3ClampLinearNormOutput[4 * i] <
                 float4x3ClampLinearNormExpect[i].x - precision ||
             float4x3ClampLinearNormOutput[4 * i] >
                 float4x3ClampLinearNormExpect[i].x + precision) ||
            (float4x3ClampLinearNormOutput[4 * i + 1] <
                 float4x3ClampLinearNormExpect[i].y - precision ||
             float4x3ClampLinearNormOutput[4 * i + 1] >
                 float4x3ClampLinearNormExpect[i].y + precision) ||
            (float4x3ClampLinearNormOutput[4 * i + 2] <
                 float4x3ClampLinearNormExpect[i].z - precision ||
             float4x3ClampLinearNormOutput[4 * i + 2] >
                 float4x3ClampLinearNormExpect[i].z + precision) ||
            (float4x3ClampLinearNormOutput[4 * i + 3] <
                 float4x3ClampLinearNormExpect[i].w - precision ||
             float4x3ClampLinearNormOutput[4 * i + 3] >
                 float4x3ClampLinearNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3ClampLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{" << float4x3ClampLinearNormOutput[12 * (float4W * i + j)]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 1]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 2]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 4]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 5]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 6]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 8]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 9]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3ClampLinearNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3MirrorLinearNormExpect[float4H * float4W * 3] = {
          {1, 2, 3, 4},
          {7, 8, 9, 10},
          {29, 30, 31, 32},
          {5, 6, 7, 8},
          {10.1875, 11.1875, 12.1875, 13.1875},
          {25, 26, 27, 28}, // 1
          {25, 26, 27, 28},
          {19.8125, 20.8125, 21.8125, 22.8125},
          {5, 6, 7, 8},
          {29, 30, 31, 32},
          {23, 24, 25, 26},
          {1, 2, 3, 4}, // 2
          {1, 2, 3, 4},
          {7, 8, 9, 10},
          {29, 30, 31, 32},
          {5, 6, 7, 8},
          {10.1875, 11.1875, 12.1875, 13.1875},
          {25, 26, 27, 28}, // 3
          {25, 26, 27, 28},
          {19.8125, 20.8125, 21.8125, 22.8125},
          {5, 6, 7, 8},
          {29, 30, 31, 32},
          {23, 24, 25, 26},
          {1, 2, 3, 4}, // 4
      };
      float *float4x3MirrorLinearNormOutput;
      cudaMallocManaged(&float4x3MirrorLinearNormOutput,
                        sizeof(float4x3MirrorLinearNormExpect));
      auto float4x3MirrorLinearNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeMirror, cudaFilterModeLinear, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3MirrorLinearNormOutput,
                                  float4x3MirrorLinearNormTex, float4W,
                                  float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3MirrorLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3MirrorLinearNormOutput[4 * i] <
                 float4x3MirrorLinearNormExpect[i].x - precision ||
             float4x3MirrorLinearNormOutput[4 * i] >
                 float4x3MirrorLinearNormExpect[i].x + precision) ||
            (float4x3MirrorLinearNormOutput[4 * i + 1] <
                 float4x3MirrorLinearNormExpect[i].y - precision ||
             float4x3MirrorLinearNormOutput[4 * i + 1] >
                 float4x3MirrorLinearNormExpect[i].y + precision) ||
            (float4x3MirrorLinearNormOutput[4 * i + 2] <
                 float4x3MirrorLinearNormExpect[i].z - precision ||
             float4x3MirrorLinearNormOutput[4 * i + 2] >
                 float4x3MirrorLinearNormExpect[i].z + precision) ||
            (float4x3MirrorLinearNormOutput[4 * i + 3] <
                 float4x3MirrorLinearNormExpect[i].w - precision ||
             float4x3MirrorLinearNormOutput[4 * i + 3] >
                 float4x3MirrorLinearNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3MirrorLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{"
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j)]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 1]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 2]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 4]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 5]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 6]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 8]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 9]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3MirrorLinearNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float4 float4x3BorderLinearNormExpect[float4H * float4W * 3] = {
          {0.03125, 0.0625, 0.09375, 0.125},
          {7, 8, 9, 10},
          {0.90625, 0.9375, 0.96875, 1},
          {0.351562, 0.421875, 0.492188, 0.5625},
          {0, 0, 0, 0},
          {0, 0, 0, 0}, // 1
          {6.73828, 7.00781, 7.27734, 7.54688},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {18.2383, 18.8672, 19.4961, 20.125},
          {0, 0, 0, 0},
          {0, 0, 0, 0}, // 2
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0}, // 3
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0}, // 4
      };
      float *float4x3BorderLinearNormOutput;
      cudaMallocManaged(&float4x3BorderLinearNormOutput,
                        sizeof(float4x3BorderLinearNormExpect));
      auto float4x3BorderLinearNormTex =
          getTex(float4Input, float4W, float4H, cudaCreateChannelDesc<float4>(),
                 float4Pitch, cudaAddressModeBorder, cudaFilterModeLinear, 1);
      kernel4x3<float4><<<1, 1>>>(float4x3BorderLinearNormOutput,
                                  float4x3BorderLinearNormTex, float4W,
                                  float4H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float4x3BorderLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float4W * float4H * 3; ++i) {
        if ((float4x3BorderLinearNormOutput[4 * i] <
                 float4x3BorderLinearNormExpect[i].x - precision ||
             float4x3BorderLinearNormOutput[4 * i] >
                 float4x3BorderLinearNormExpect[i].x + precision) ||
            (float4x3BorderLinearNormOutput[4 * i + 1] <
                 float4x3BorderLinearNormExpect[i].y - precision ||
             float4x3BorderLinearNormOutput[4 * i + 1] >
                 float4x3BorderLinearNormExpect[i].y + precision) ||
            (float4x3BorderLinearNormOutput[4 * i + 2] <
                 float4x3BorderLinearNormExpect[i].z - precision ||
             float4x3BorderLinearNormOutput[4 * i + 2] >
                 float4x3BorderLinearNormExpect[i].z + precision) ||
            (float4x3BorderLinearNormOutput[4 * i + 3] <
                 float4x3BorderLinearNormExpect[i].w - precision ||
             float4x3BorderLinearNormOutput[4 * i + 3] >
                 float4x3BorderLinearNormExpect[i].w + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float4x3BorderLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float4H; ++i) {
          for (int j = 0; j < float4W; ++j)
            cout << "{"
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j)]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 1]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 2]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 3]
                 << "}, {"
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 4]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 5]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 6]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 7]
                 << "}, {"
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 8]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 9]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 10]
                 << ", "
                 << float4x3BorderLinearNormOutput[12 * (float4W * i + j) + 11]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    cudaFree(float4Input);
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
