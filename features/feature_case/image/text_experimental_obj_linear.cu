// ===-------- text_experimental_obj_linear.cu ------- *- CUDA -* ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

#define PRINT_PASS 0

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
__global__ void kernel1(EleT *output, cudaTextureObject_t tex, int num) {
  for (int i = 0; i < num; ++i) {
    auto ret = tex1Dfetch<T>(tex, i);
    output[i] = ret.x;
  }
}

template <typename T, typename EleT>
__global__ void kernel2(EleT *output, cudaTextureObject_t tex, int num) {
  for (int i = 0; i < num; ++i) {
    auto ret = tex1Dfetch<T>(tex, i);
    output[2 * i] = ret.x;
    output[2 * i + 1] = ret.y;
  }
}

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaTextureObject_t tex, int num) {
  for (int i = 0; i < num; ++i) {
    auto ret = tex1Dfetch<T>(tex, i);
    output[4 * i] = ret.x;
    output[4 * i + 1] = ret.y;
    output[4 * i + 2] = ret.z;
    output[4 * i + 3] = ret.w;
  }
}

template <typename T, typename ArrT> T *getInput(ArrT &expect) {
  T *input;
  cudaMalloc(&input, sizeof(expect));
  cudaMemcpy(input, &expect, sizeof(expect), cudaMemcpyHostToDevice);
  return input;
}

cudaTextureObject_t getTex(void *input, cudaChannelFormatDesc desc,
                           size_t sizeInBytes) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = input;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = sizeInBytes;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return tex;
}

int main() {
  bool pass = true;

  {
    const int char1Num = 6;
    char1 char1Expect[char1Num] = {
        1, 2, 3, 4, 5, 6,
    };
    auto *char1Input = getInput<char1>(char1Expect);
    char *char1Output;
    cudaMallocManaged(&char1Output, sizeof(char1Expect));
    auto char1Tex =
        getTex(char1Input, cudaCreateChannelDesc<char1>(), sizeof(char1Expect));
    kernel1<char1><<<1, 1>>>(char1Output, char1Tex, char1Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(char1Tex);
    cudaFree(char1Input);
    for (int i = 0; i < char1Num; ++i) {
      if (char1Output[i] != char1Expect[i].x) {
        pass = false;
        break;
      }
    }
    checkResult("char1", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < char1Num; ++i)
        cout << "{" << (long)char1Output[i] << "}" << endl;
    pass = true;
  }

  {
    const int uchar2Num = 5;
    uchar2 uchar2Expect[uchar2Num] = {
        {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
    };
    auto *uchar2Input = getInput<uchar2>(uchar2Expect);
    unsigned char *uchar2Output;
    cudaMallocManaged(&uchar2Output, sizeof(uchar2Expect));
    auto uchar2Tex = getTex(uchar2Input, cudaCreateChannelDesc<uchar2>(),
                            sizeof(uchar2Expect));
    kernel2<uchar2><<<1, 1>>>(uchar2Output, uchar2Tex, uchar2Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(uchar2Tex);
    cudaFree(uchar2Input);
    for (int i = 0; i < uchar2Num; ++i) {
      if (uchar2Output[2 * i] != uchar2Expect[i].x ||
          uchar2Output[2 * i + 1] != uchar2Expect[i].y) {
        pass = false;
        break;
      }
    }
    checkResult("uchar2", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < uchar2Num; ++i)
        cout << "{" << (long)uchar2Output[2 * i] << ", "
             << (long)uchar2Output[2 * i + 1] << "}" << endl;
    pass = true;
  }

  {
    const int short2Num = 3;
    short2 short2Expect[short2Num] = {
        {1, 2},
        {3, 4},
        {5, 6},
    };
    auto *short2Input = getInput<short2>(short2Expect);
    short *short2Output;
    cudaMallocManaged(&short2Output, sizeof(short2Expect));
    auto short2Tex = getTex(short2Input, cudaCreateChannelDesc<short2>(),
                            sizeof(short2Expect));
    kernel2<short2><<<1, 1>>>(short2Output, short2Tex, short2Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(short2Tex);
    cudaFree(short2Input);
    for (int i = 0; i < short2Num; ++i) {
      if (short2Output[2 * i] != short2Expect[i].x ||
          short2Output[2 * i + 1] != short2Expect[i].y) {
        pass = false;
        break;
      }
    }
    checkResult("short2", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < short2Num; ++i)
        cout << "{" << (long)short2Output[2 * i] << ", "
             << (long)short2Output[2 * i + 1] << "}" << endl;
    pass = true;
  }

  {
    const int ushort4Num = 2;
    ushort4 ushort4Expect[ushort4Num] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    };
    auto *ushort4Input = getInput<ushort4>(ushort4Expect);
    unsigned short *ushort4Output;
    cudaMallocManaged(&ushort4Output, sizeof(ushort4Expect));
    auto ushort4Tex = getTex(ushort4Input, cudaCreateChannelDesc<ushort4>(),
                             sizeof(ushort4Expect));
    kernel4<ushort4><<<1, 1>>>(ushort4Output, ushort4Tex, ushort4Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(ushort4Tex);
    cudaFree(ushort4Input);
    for (int i = 0; i < ushort4Num; ++i) {
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
      for (int i = 0; i < ushort4Num; ++i)
        cout << "{" << (long)ushort4Output[4 * i] << ", "
             << (long)ushort4Output[4 * i + 1] << ", "
             << (long)ushort4Output[4 * i + 2] << ", "
             << (long)ushort4Output[4 * i + 3] << "}" << endl;
    pass = true;
  }

  {
    const int int1Num = 5;
    int1 int1Expect[int1Num] = {
        1, 2, 3, 4, 5,
    };
    auto *int1Input = getInput<int1>(int1Expect);
    int *int1Output;
    cudaMallocManaged(&int1Output, sizeof(int1Expect));
    auto int1Tex =
        getTex(int1Input, cudaCreateChannelDesc<int1>(), sizeof(int1Expect));
    kernel1<int1><<<1, 1>>>(int1Output, int1Tex, int1Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(int1Tex);
    cudaFree(int1Input);
    for (int i = 0; i < int1Num; ++i) {
      if (int1Output[i] != int1Expect[i].x) {
        pass = false;
        break;
      }
    }
    checkResult("int1", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < int1Num; ++i)
        cout << "{" << (long)int1Output[i] << "}" << endl;
    pass = true;
  }

  {
    const int uint4Num = 3;
    uint4 uint4Expect[uint4Num] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
    };
    auto *uint4Input = getInput<uint4>(uint4Expect);
    unsigned int *uint4Output;
    cudaMallocManaged(&uint4Output, sizeof(uint4Expect));
    auto uint4Tex =
        getTex(uint4Input, cudaCreateChannelDesc<uint4>(), sizeof(uint4Expect));
    kernel4<uint4><<<1, 1>>>(uint4Output, uint4Tex, uint4Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(uint4Tex);
    cudaFree(uint4Input);
    for (int i = 0; i < uint4Num; ++i) {
      if (uint4Output[4 * i] != uint4Expect[i].x ||
          uint4Output[4 * i + 1] != uint4Expect[i].y ||
          uint4Output[4 * i + 2] != uint4Expect[i].z ||
          uint4Output[4 * i + 3] != uint4Expect[i].w) {
        pass = false;
        break;
      }
    }
    checkResult("uint4", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < uint4Num; ++i)
        cout << "{" << (long)uint4Output[4 * i] << ", "
             << (long)uint4Output[4 * i + 1] << ", "
             << (long)uint4Output[4 * i + 2] << ", "
             << (long)uint4Output[4 * i + 3] << "}" << endl;
    pass = true;
  }

  {
    const int float4Num = 4;
    float4 float4Expect[float4Num] = {
        {0.5, 1, 1.5, 3},
        {4.5, 5, 5.5, 7},
        {8.5, 9, 9.5, 11},
        {12.5, 13, 13.5, 15},
    };
    auto *float4Input = getInput<float4>(float4Expect);
    float *float4Output;
    cudaMallocManaged(&float4Output, sizeof(float4Expect));
    auto float4Tex = getTex(float4Input, cudaCreateChannelDesc<float4>(),
                            sizeof(float4Expect));
    kernel4<float4><<<1, 1>>>(float4Output, float4Tex, float4Num);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(float4Tex);
    cudaFree(float4Input);
    float precision = 0.001;
    for (int i = 0; i < float4Num; ++i) {
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
      for (int i = 0; i < float4Num; ++i)
        cout << "{" << float4Output[4 * i] << ", " << float4Output[4 * i + 1]
             << ", " << float4Output[4 * i + 2] << ", "
             << float4Output[4 * i + 3] << "}" << endl;
    pass = true;
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
