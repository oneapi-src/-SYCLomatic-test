// ===-------- text_experimental_obj_array.cu ------- *- CUDA -* ----------===//
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
__global__ void kernel2x3(EleT *output, cudaTextureObject_t tex, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = tex2D<T>(tex, j - 0.1, i - 0.1);
      output[6 * (w * i + j)] = ret.x;
      output[6 * (w * i + j) + 1] = ret.y;
      auto ret1 = tex2D<T>(tex, j + 0.3, i + 0.3);
      output[6 * (w * i + j) + 2] = ret1.x;
      output[6 * (w * i + j) + 3] = ret1.y;
      auto ret2 = tex2D<T>(tex, j + 1.1, i + 1.1);
      output[6 * (w * i + j) + 4] = ret2.x;
      output[6 * (w * i + j) + 5] = ret2.y;
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
    auto *char2Input = getInput<char2>(char2Expect, char2W, char2H,
                                       cudaCreateChannelDesc<char2>());
    char *char2Output;
    cudaMallocManaged(&char2Output, sizeof(char2Expect));
    auto char2Tex = getTex(char2Input);
    kernel2<char2><<<1, 1>>>(char2Output, char2Tex, char2W, char2H);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(char2Tex);
    cudaFreeArray(char2Input);
    for (int i = 0; i < char2W * char2H; ++i) {
      if (char2Output[2 * i] != char2Expect[i].x ||
          char2Output[2 * i + 1] != char2Expect[i].y) {
        pass = false;
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
    const int short4H = 2;
    const int short4W = 4;
    short4 short4Expect[short4H * short4W] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 1
        {17, 18, 19, 20}, {21, 22, 23, 24},
        {25, 26, 27, 28}, {29, 30, 31, 32}, // 2
    };
    auto *short4Input = getInput<short4>(short4Expect, short4W, short4H,
                                         cudaCreateChannelDesc<short4>());
    short *short4Output;
    cudaMallocManaged(&short4Output, sizeof(short4Expect));
    auto short4Tex = getTex(short4Input);
    kernel4<short4><<<1, 1>>>(short4Output, short4Tex, short4W, short4H);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(short4Tex);
    cudaFreeArray(short4Input);
    for (int i = 0; i < short4W * short4H; ++i) {
      if (short4Output[4 * i] != short4Expect[i].x ||
          short4Output[4 * i + 1] != short4Expect[i].y ||
          short4Output[4 * i + 2] != short4Expect[i].z ||
          short4Output[4 * i + 3] != short4Expect[i].w) {
        pass = false;
        break;
      }
    }
    checkResult("short4", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < short4H; ++i) {
        for (int j = 0; j < short4W; ++j)
          cout << "{" << short4Output[4 * (short4W * i + j)] << ", "
               << short4Output[4 * (short4W * i + j) + 1] << ", "
               << short4Output[4 * (short4W * i + j) + 2] << ", "
               << short4Output[4 * (short4W * i + j) + 3] << "}, ";
        cout << endl;
      }
    pass = true;
  }

  {
    const int float2H = 2;
    const int float2W = 4;
    float2 float2Expect[float2H * float2W] = {
        {1, 2},  {3, 4},   {5, 6},   {7, 8},   // 1
        {9, 10}, {11, 12}, {13, 14}, {15, 16}, // 2
    };
    auto *float2Input = getInput<float2>(float2Expect, float2W, float2H,
                                         cudaCreateChannelDesc<float2>());
    {
      short *float2Output;
      cudaMallocManaged(&float2Output, sizeof(float2Expect));
      auto float2Tex = getTex(float2Input);
      kernel2<float2><<<1, 1>>>(float2Output, float2Tex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2Tex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H; ++i) {
        if ((float2Output[2 * i] < float2Expect[i].x - precision ||
             float2Output[2 * i] > float2Expect[i].x + precision) ||
            (float2Output[2 * i + 1] < float2Expect[i].y - precision ||
             float2Output[2 * i + 1] > float2Expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2Output[2 * (float2W * i + j)] << ", "
                 << float2Output[2 * (float2W * i + j) + 1] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3Expect[float2H * float2W * 3] = {
          {1, 2}, {1, 2},   {11, 12}, {1, 2}, {3, 4},   {13, 14},
          {3, 4}, {5, 6},   {15, 16}, {5, 6}, {7, 8},   {15, 16}, // 1
          {1, 2}, {9, 10},  {11, 12}, {1, 2}, {11, 12}, {13, 14},
          {3, 4}, {13, 14}, {15, 16}, {5, 6}, {15, 16}, {15, 16}, // 2
      };
      short *float2x3Output;
      cudaMallocManaged(&float2x3Output, sizeof(float2x3Expect));
      auto float2x3Tex = getTex(float2Input);
      kernel2x3<float2>
          <<<1, 1>>>(float2x3Output, float2x3Tex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3Tex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3Output[2 * i] < float2x3Expect[i].x - precision ||
             float2x3Output[2 * i] > float2x3Expect[i].x + precision) ||
            (float2x3Output[2 * i + 1] < float2x3Expect[i].y - precision ||
             float2x3Output[2 * i + 1] > float2x3Expect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3Output[6 * (float2W * i + j)] << ", "
                 << float2x3Output[6 * (float2W * i + j) + 1] << "}, {"
                 << float2x3Output[6 * (float2W * i + j) + 2] << ", "
                 << float2x3Output[6 * (float2W * i + j) + 3] << "}, {"
                 << float2x3Output[6 * (float2W * i + j) + 4] << ", "
                 << float2x3Output[6 * (float2W * i + j) + 5] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3BorderExpect[float2H * float2W * 3] = {
          {0, 0}, {1, 2},   {11, 12}, {0, 0}, {3, 4},   {13, 14},
          {0, 0}, {5, 6},   {15, 16}, {0, 0}, {7, 8},   {0, 0}, // 1
          {0, 0}, {9, 10},  {0, 0},   {1, 2}, {11, 12}, {0, 0},
          {3, 4}, {13, 14}, {0, 0},   {5, 6}, {15, 16}, {0, 0}, // 2
      };
      short *float2x3BorderOutput;
      cudaMallocManaged(&float2x3BorderOutput, sizeof(float2x3BorderExpect));
      auto float2x3BorderTex = getTex(float2Input, cudaAddressModeBorder);
      kernel2x3<float2>
          <<<1, 1>>>(float2x3BorderOutput, float2x3BorderTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3BorderTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3BorderOutput[2 * i] <
                 float2x3BorderExpect[i].x - precision ||
             float2x3BorderOutput[2 * i] >
                 float2x3BorderExpect[i].x + precision) ||
            (float2x3BorderOutput[2 * i + 1] <
                 float2x3BorderExpect[i].y - precision ||
             float2x3BorderOutput[2 * i + 1] >
                 float2x3BorderExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3Border", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3BorderOutput[6 * (float2W * i + j)] << ", "
                 << float2x3BorderOutput[6 * (float2W * i + j) + 1] << "}, {"
                 << float2x3BorderOutput[6 * (float2W * i + j) + 2] << ", "
                 << float2x3BorderOutput[6 * (float2W * i + j) + 3] << "}, {"
                 << float2x3BorderOutput[6 * (float2W * i + j) + 4] << ", "
                 << float2x3BorderOutput[6 * (float2W * i + j) + 5] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3WrapLinearExpect[float2H * float2W * 3] = {
          {1, 2}, {1, 2},   {7, 8},   {1, 2}, {2, 3},   {9, 10},
          {3, 4}, {4, 5},   {11, 12}, {5, 6}, {6, 7},   {11, 12}, // 1
          {4, 5}, {7, 8},   {10, 11}, {4, 5}, {9, 10},  {12, 13},
          {6, 7}, {11, 12}, {14, 15}, {8, 9}, {13, 14}, {15, 16}, // 2
      };
      short *float2x3WrapLinearOutput;
      cudaMallocManaged(&float2x3WrapLinearOutput,
                        sizeof(float2x3WrapLinearExpect));
      auto float2x3WrapLinearTex =
          getTex(float2Input, cudaAddressModeWrap, cudaFilterModeLinear);
      kernel2x3<float2><<<1, 1>>>(float2x3WrapLinearOutput,
                                  float2x3WrapLinearTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3WrapLinearTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3WrapLinearOutput[2 * i] <
                 float2x3WrapLinearExpect[i].x - precision ||
             float2x3WrapLinearOutput[2 * i] >
                 float2x3WrapLinearExpect[i].x + precision) ||
            (float2x3WrapLinearOutput[2 * i + 1] <
                 float2x3WrapLinearExpect[i].y - precision ||
             float2x3WrapLinearOutput[2 * i + 1] >
                 float2x3WrapLinearExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3WrapLinear", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3WrapLinearOutput[6 * (float2W * i + j)]
                 << ", " << float2x3WrapLinearOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3WrapLinearOutput[6 * (float2W * i + j) + 2] << ", "
                 << float2x3WrapLinearOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3WrapLinearOutput[6 * (float2W * i + j) + 4] << ", "
                 << float2x3WrapLinearOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3BorderLinearExpect[float2H * float2W * 3] = {
          {0, 0}, {0, 1},   {7, 8},   {0, 1}, {2, 2},   {9, 10},
          {1, 1}, {3, 4},   {11, 12}, {2, 2}, {5, 6},   {4, 5}, // 1
          {1, 2}, {5, 6},   {4, 4},   {4, 5}, {9, 10},  {4, 5},
          {6, 7}, {11, 12}, {5, 6},   {8, 9}, {13, 14}, {2, 2}, // 2
      };
      short *float2x3BorderLinearOutput;
      cudaMallocManaged(&float2x3BorderLinearOutput,
                        sizeof(float2x3BorderLinearExpect));
      auto float2x3BorderLinearTex =
          getTex(float2Input, cudaAddressModeBorder, cudaFilterModeLinear);
      kernel2x3<float2><<<1, 1>>>(float2x3BorderLinearOutput,
                                  float2x3BorderLinearTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3BorderLinearTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3BorderLinearOutput[2 * i] <
                 float2x3BorderLinearExpect[i].x - precision ||
             float2x3BorderLinearOutput[2 * i] >
                 float2x3BorderLinearExpect[i].x + precision) ||
            (float2x3BorderLinearOutput[2 * i + 1] <
                 float2x3BorderLinearExpect[i].y - precision ||
             float2x3BorderLinearOutput[2 * i + 1] >
                 float2x3BorderLinearExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3BorderLinear", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3BorderLinearOutput[6 * (float2W * i + j)]
                 << ", "
                 << float2x3BorderLinearOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3BorderLinearOutput[6 * (float2W * i + j) + 2]
                 << ", "
                 << float2x3BorderLinearOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3BorderLinearOutput[6 * (float2W * i + j) + 4]
                 << ", "
                 << float2x3BorderLinearOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3WrapNormExpect[float2H * float2W * 3] = {
          {15, 16}, {3, 4}, {1, 2}, {15, 16}, {3, 4}, {1, 2},
          {15, 16}, {3, 4}, {1, 2}, {15, 16}, {3, 4}, {1, 2}, // 1
          {15, 16}, {3, 4}, {1, 2}, {15, 16}, {3, 4}, {1, 2},
          {15, 16}, {3, 4}, {1, 2}, {15, 16}, {3, 4}, {1, 2}, // 2
      };
      short *float2x3WrapNormOutput;
      cudaMallocManaged(&float2x3WrapNormOutput,
                        sizeof(float2x3WrapNormExpect));
      auto float2x3WrapNormTex =
          getTex(float2Input, cudaAddressModeWrap, cudaFilterModePoint, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3WrapNormOutput, float2x3WrapNormTex,
                                  float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3WrapNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3WrapNormOutput[2 * i] <
                 float2x3WrapNormExpect[i].x - precision ||
             float2x3WrapNormOutput[2 * i] >
                 float2x3WrapNormExpect[i].x + precision) ||
            (float2x3WrapNormOutput[2 * i + 1] <
                 float2x3WrapNormExpect[i].y - precision ||
             float2x3WrapNormOutput[2 * i + 1] >
                 float2x3WrapNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3WrapNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3WrapNormOutput[6 * (float2W * i + j)] << ", "
                 << float2x3WrapNormOutput[6 * (float2W * i + j) + 1] << "}, {"
                 << float2x3WrapNormOutput[6 * (float2W * i + j) + 2] << ", "
                 << float2x3WrapNormOutput[6 * (float2W * i + j) + 3] << "}, {"
                 << float2x3WrapNormOutput[6 * (float2W * i + j) + 4] << ", "
                 << float2x3WrapNormOutput[6 * (float2W * i + j) + 5] << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3ClampNormExpect[float2H * float2W * 3] = {
          {1, 2},   {3, 4},   {15, 16}, {7, 8},   {7, 8},   {15, 16},
          {7, 8},   {7, 8},   {15, 16}, {7, 8},   {7, 8},   {15, 16}, // 1
          {9, 10},  {11, 12}, {15, 16}, {15, 16}, {15, 16}, {15, 16},
          {15, 16}, {15, 16}, {15, 16}, {15, 16}, {15, 16}, {15, 16}, // 2
      };
      short *float2x3ClampNormOutput;
      cudaMallocManaged(&float2x3ClampNormOutput,
                        sizeof(float2x3ClampNormExpect));
      auto float2x3ClampNormTex =
          getTex(float2Input, cudaAddressModeClamp, cudaFilterModePoint, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3ClampNormOutput, float2x3ClampNormTex,
                                  float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3ClampNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3ClampNormOutput[2 * i] <
                 float2x3ClampNormExpect[i].x - precision ||
             float2x3ClampNormOutput[2 * i] >
                 float2x3ClampNormExpect[i].x + precision) ||
            (float2x3ClampNormOutput[2 * i + 1] <
                 float2x3ClampNormExpect[i].y - precision ||
             float2x3ClampNormOutput[2 * i + 1] >
                 float2x3ClampNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3ClampNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3ClampNormOutput[6 * (float2W * i + j)]
                 << ", " << float2x3ClampNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {" << float2x3ClampNormOutput[6 * (float2W * i + j) + 2]
                 << ", " << float2x3ClampNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {" << float2x3ClampNormOutput[6 * (float2W * i + j) + 4]
                 << ", " << float2x3ClampNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3MirrorNormExpect[float2H * float2W * 3] = {
          {1, 2},  {3, 4},   {15, 16}, {7, 8},   {5, 6},   {9, 10},
          {1, 2},  {3, 4},   {15, 16}, {7, 8},   {5, 6},   {9, 10}, // 1
          {9, 10}, {11, 12}, {7, 8},   {15, 16}, {13, 14}, {1, 2},
          {9, 10}, {11, 12}, {7, 8},   {15, 16}, {13, 14}, {1, 2}, // 2
      };
      short *float2x3MirrorNormOutput;
      cudaMallocManaged(&float2x3MirrorNormOutput,
                        sizeof(float2x3MirrorNormExpect));
      auto float2x3MirrorNormTex =
          getTex(float2Input, cudaAddressModeMirror, cudaFilterModePoint, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3MirrorNormOutput,
                                  float2x3MirrorNormTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3MirrorNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3MirrorNormOutput[2 * i] <
                 float2x3MirrorNormExpect[i].x - precision ||
             float2x3MirrorNormOutput[2 * i] >
                 float2x3MirrorNormExpect[i].x + precision) ||
            (float2x3MirrorNormOutput[2 * i + 1] <
                 float2x3MirrorNormExpect[i].y - precision ||
             float2x3MirrorNormOutput[2 * i + 1] >
                 float2x3MirrorNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3MirrorNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3MirrorNormOutput[6 * (float2W * i + j)]
                 << ", " << float2x3MirrorNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3MirrorNormOutput[6 * (float2W * i + j) + 2] << ", "
                 << float2x3MirrorNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3MirrorNormOutput[6 * (float2W * i + j) + 4] << ", "
                 << float2x3MirrorNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3BorderNormExpect[float2H * float2W * 3] = {
          {0, 0}, {3, 4}, {0, 0}, {0, 0},   {0, 0}, {0, 0},
          {0, 0}, {0, 0}, {0, 0}, {0, 0},   {0, 0}, {0, 0}, // 1
          {0, 0}, {0, 0}, {0, 0}, {15, 16}, {0, 0}, {0, 0},
          {0, 0}, {0, 0}, {0, 0}, {0, 0},   {0, 0}, {0, 0}, // 2
      };
      short *float2x3BorderNormOutput;
      cudaMallocManaged(&float2x3BorderNormOutput,
                        sizeof(float2x3BorderNormExpect));
      auto float2x3BorderNormTex =
          getTex(float2Input, cudaAddressModeBorder, cudaFilterModePoint, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3BorderNormOutput,
                                  float2x3BorderNormTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3BorderNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3BorderNormOutput[2 * i] <
                 float2x3BorderNormExpect[i].x - precision ||
             float2x3BorderNormOutput[2 * i] >
                 float2x3BorderNormExpect[i].x + precision) ||
            (float2x3BorderNormOutput[2 * i + 1] <
                 float2x3BorderNormExpect[i].y - precision ||
             float2x3BorderNormOutput[2 * i + 1] >
                 float2x3BorderNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3BorderNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3BorderNormOutput[6 * (float2W * i + j)]
                 << ", " << float2x3BorderNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3BorderNormOutput[6 * (float2W * i + j) + 2] << ", "
                 << float2x3BorderNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3BorderNormOutput[6 * (float2W * i + j) + 4] << ", "
                 << float2x3BorderNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3WrapLinearNormExpect[float2H * float2W * 3] = {
          {11, 12}, {3, 4}, {4, 5}, {11, 12}, {3, 4}, {4, 5},
          {11, 12}, {3, 4}, {4, 5}, {11, 12}, {3, 4}, {4, 5}, // 1
          {11, 12}, {3, 4}, {4, 5}, {11, 12}, {3, 4}, {4, 5},
          {11, 12}, {3, 4}, {4, 5}, {11, 12}, {3, 4}, {4, 5}, // 2
      };
      short *float2x3WrapLinearNormOutput;
      cudaMallocManaged(&float2x3WrapLinearNormOutput,
                        sizeof(float2x3WrapLinearNormExpect));
      auto float2x3WrapLinearNormTex =
          getTex(float2Input, cudaAddressModeWrap, cudaFilterModeLinear, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3WrapLinearNormOutput,
                                  float2x3WrapLinearNormTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3WrapLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3WrapLinearNormOutput[2 * i] <
                 float2x3WrapLinearNormExpect[i].x - precision ||
             float2x3WrapLinearNormOutput[2 * i] >
                 float2x3WrapLinearNormExpect[i].x + precision) ||
            (float2x3WrapLinearNormOutput[2 * i + 1] <
                 float2x3WrapLinearNormExpect[i].y - precision ||
             float2x3WrapLinearNormOutput[2 * i + 1] >
                 float2x3WrapLinearNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3WrapLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3WrapLinearNormOutput[6 * (float2W * i + j)]
                 << ", "
                 << float2x3WrapLinearNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3WrapLinearNormOutput[6 * (float2W * i + j) + 2]
                 << ", "
                 << float2x3WrapLinearNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3WrapLinearNormOutput[6 * (float2W * i + j) + 4]
                 << ", "
                 << float2x3WrapLinearNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3ClampLinearNormExpect[float2H * float2W * 3] = {
          {1, 2},   {3, 4},   {15, 16}, {7, 8},   {7, 8},   {15, 16},
          {7, 8},   {7, 8},   {15, 16}, {7, 8},   {7, 8},   {15, 16}, // 1
          {9, 10},  {10, 11}, {15, 16}, {15, 16}, {15, 16}, {15, 16},
          {15, 16}, {15, 16}, {15, 16}, {15, 16}, {15, 16}, {15, 16}, // 2
      };
      short *float2x3ClampLinearNormOutput;
      cudaMallocManaged(&float2x3ClampLinearNormOutput,
                        sizeof(float2x3ClampLinearNormExpect));
      auto float2x3ClampLinearNormTex =
          getTex(float2Input, cudaAddressModeClamp, cudaFilterModeLinear, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3ClampLinearNormOutput,
                                  float2x3ClampLinearNormTex, float2W, float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3ClampLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3ClampLinearNormOutput[2 * i] <
                 float2x3ClampLinearNormExpect[i].x - precision ||
             float2x3ClampLinearNormOutput[2 * i] >
                 float2x3ClampLinearNormExpect[i].x + precision) ||
            (float2x3ClampLinearNormOutput[2 * i + 1] <
                 float2x3ClampLinearNormExpect[i].y - precision ||
             float2x3ClampLinearNormOutput[2 * i + 1] >
                 float2x3ClampLinearNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3ClampLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3ClampLinearNormOutput[6 * (float2W * i + j)]
                 << ", "
                 << float2x3ClampLinearNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3ClampLinearNormOutput[6 * (float2W * i + j) + 2]
                 << ", "
                 << float2x3ClampLinearNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3ClampLinearNormOutput[6 * (float2W * i + j) + 4]
                 << ", "
                 << float2x3ClampLinearNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3MirrorLinearNormExpect[float2H * float2W * 3] = {
          {1, 2},  {3, 4},  {15, 16}, {7, 8},   {6, 7},   {9, 10},
          {1, 2},  {3, 4},  {15, 16}, {7, 8},   {6, 7},   {9, 10}, // 1
          {9, 10}, {9, 10}, {7, 8},   {15, 16}, {12, 13}, {1, 2},
          {9, 10}, {9, 10}, {7, 8},   {15, 16}, {12, 13}, {1, 2}, // 2
      };
      short *float2x3MirrorLinearNormOutput;
      cudaMallocManaged(&float2x3MirrorLinearNormOutput,
                        sizeof(float2x3MirrorLinearNormExpect));
      auto float2x3MirrorLinearNormTex =
          getTex(float2Input, cudaAddressModeMirror, cudaFilterModeLinear, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3MirrorLinearNormOutput,
                                  float2x3MirrorLinearNormTex, float2W,
                                  float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3MirrorLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3MirrorLinearNormOutput[2 * i] <
                 float2x3MirrorLinearNormExpect[i].x - precision ||
             float2x3MirrorLinearNormOutput[2 * i] >
                 float2x3MirrorLinearNormExpect[i].x + precision) ||
            (float2x3MirrorLinearNormOutput[2 * i + 1] <
                 float2x3MirrorLinearNormExpect[i].y - precision ||
             float2x3MirrorLinearNormOutput[2 * i + 1] >
                 float2x3MirrorLinearNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3MirrorLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3MirrorLinearNormOutput[6 * (float2W * i + j)]
                 << ", "
                 << float2x3MirrorLinearNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3MirrorLinearNormOutput[6 * (float2W * i + j) + 2]
                 << ", "
                 << float2x3MirrorLinearNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3MirrorLinearNormOutput[6 * (float2W * i + j) + 4]
                 << ", "
                 << float2x3MirrorLinearNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    {
      float2 float2x3BorderLinearNormExpect[float2H * float2W * 3] = {
          {0, 0}, {3, 4}, {0, 0}, {1, 2},  {0, 0}, {0, 0},
          {0, 0}, {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0}, // 1
          {0, 0}, {0, 0}, {0, 0}, {9, 10}, {0, 0}, {0, 0},
          {0, 0}, {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0}, // 2
      };
      short *float2x3BorderLinearNormOutput;
      cudaMallocManaged(&float2x3BorderLinearNormOutput,
                        sizeof(float2x3BorderLinearNormExpect));
      auto float2x3BorderLinearNormTex =
          getTex(float2Input, cudaAddressModeBorder, cudaFilterModeLinear, 1);
      kernel2x3<float2><<<1, 1>>>(float2x3BorderLinearNormOutput,
                                  float2x3BorderLinearNormTex, float2W,
                                  float2H);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(float2x3BorderLinearNormTex);
      float precision = 0.001;
      for (int i = 0; i < float2W * float2H * 3; ++i) {
        if ((float2x3BorderLinearNormOutput[2 * i] <
                 float2x3BorderLinearNormExpect[i].x - precision ||
             float2x3BorderLinearNormOutput[2 * i] >
                 float2x3BorderLinearNormExpect[i].x + precision) ||
            (float2x3BorderLinearNormOutput[2 * i + 1] <
                 float2x3BorderLinearNormExpect[i].y - precision ||
             float2x3BorderLinearNormOutput[2 * i + 1] >
                 float2x3BorderLinearNormExpect[i].y + precision)) {
          pass = false;
          break;
        }
      }
      checkResult("float2x3BorderLinearNorm", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < float2H; ++i) {
          for (int j = 0; j < float2W; ++j)
            cout << "{" << float2x3BorderLinearNormOutput[6 * (float2W * i + j)]
                 << ", "
                 << float2x3BorderLinearNormOutput[6 * (float2W * i + j) + 1]
                 << "}, {"
                 << float2x3BorderLinearNormOutput[6 * (float2W * i + j) + 2]
                 << ", "
                 << float2x3BorderLinearNormOutput[6 * (float2W * i + j) + 3]
                 << "}, {"
                 << float2x3BorderLinearNormOutput[6 * (float2W * i + j) + 4]
                 << ", "
                 << float2x3BorderLinearNormOutput[6 * (float2W * i + j) + 5]
                 << "}, ";
          cout << endl;
        }
      pass = true;
    }
    cudaFreeArray(float2Input);
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
