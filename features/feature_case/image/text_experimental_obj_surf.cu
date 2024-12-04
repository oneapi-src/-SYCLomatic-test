// ===-------- text_experimental_obj_surf.cu ------- *- CUDA -* ----------===//
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
__global__ void kernel1(EleT *output, cudaSurfaceObject_t surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = surf1Dread<T>(surf, i * sizeof(T));
    output[i] = ret.x;
  }
}

template <typename T, typename EleT>
__global__ void kernel1(EleT *output, cudaSurfaceObject_t surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = surf2Dread<T>(surf, j * sizeof(T), i);

      output[w * i + j] = ret.x;
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel1(EleT *output, cudaSurfaceObject_t surf, int w, int h,
                        int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = surf3Dread<T>(surf, k * sizeof(T), j, i);
        output[w * h * i + w * j + k] = ret.x;
      }
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel2(EleT *output, cudaSurfaceObject_t surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = surf1Dread<T>(surf, i * sizeof(T));
    output[2 * i] = ret.x;
    output[2 * i + 1] = ret.y;
  }
}

template <typename T, typename EleT>
__global__ void kernel2(EleT *output, cudaSurfaceObject_t surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = surf2Dread<T>(surf, j * sizeof(T), i);
      output[2 * (w * i + j)] = ret.x;
      output[2 * (w * i + j) + 1] = ret.y;
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel2(EleT *output, cudaSurfaceObject_t surf, int w, int h,
                        int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = surf3Dread<T>(surf, k * sizeof(T), j, i);
        output[2 * (w * h * i + w * j + k)] = ret.x;
        output[2 * (w * h * i + w * j + k) + 1] = ret.y;
      }
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaSurfaceObject_t surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = surf1Dread<T>(surf, i * sizeof(T));
    output[4 * i] = ret.x;
    output[4 * i + 1] = ret.y;
    output[4 * i + 2] = ret.z;
    output[4 * i + 3] = ret.w;
  }
}

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaSurfaceObject_t surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = surf2Dread<T>(surf, j * sizeof(T), i);
      output[4 * (w * i + j)] = ret.x;
      output[4 * (w * i + j) + 1] = ret.y;
      output[4 * (w * i + j) + 2] = ret.z;
      output[4 * (w * i + j) + 3] = ret.w;
    }
  }
}

template <typename T, typename EleT>
__global__ void kernel4(EleT *output, cudaSurfaceObject_t surf, int w, int h,
                        int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = surf3Dread<T>(surf, k * sizeof(T), j, i);
        output[4 * (w * h * i + w * j + k)] = ret.x;
        output[4 * (w * h * i + w * j + k) + 1] = ret.y;
        output[4 * (w * h * i + w * j + k) + 2] = ret.z;
        output[4 * (w * h * i + w * j + k) + 3] = ret.w;
      }
    }
  }
}

template <typename T, typename ArrT>
cudaArray *getInput(ArrT &expect, size_t w, const cudaChannelFormatDesc &desc) {
  cudaArray *input;
  cudaMallocArray(&input, &desc, w, 0);
  cudaMemcpy2DToArray(input, 0, 0, expect, sizeof(T) * w, sizeof(T) * w,
                      1 /* Notice: need set height to 1!!! */,
                      cudaMemcpyHostToDevice);
  return input;
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

template <typename T, typename ArrT>
cudaArray *getInput(ArrT &expect, size_t w, size_t h, size_t d,
                    const cudaChannelFormatDesc &desc) {
  cudaArray *input;
  cudaMalloc3DArray(&input, &desc, {w, h, d});
  cudaMemcpy3DParms p = {};
  p.srcPtr = make_cudaPitchedPtr(expect, w * sizeof(T), w, h);
  p.dstArray = input;
  p.extent = make_cudaExtent(w, h, d);
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
  return input;
}

cudaSurfaceObject_t getSurf(cudaArray_t input) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = input;

  cudaSurfaceObject_t surf;
  cudaCreateSurfaceObject(&surf, &resDesc);

  return surf;
}

int main() {
  bool pass = true;

  { // 1 element uint 1D image.
    const int w = 8;
    uint1 expect[w] = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},
    };
    auto *input = getInput<uint1>(expect, w, cudaCreateChannelDesc<uint1>());
    unsigned int *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel1<uint1><<<1, 1>>>(output, surf, w);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w; ++i) {
      if (output[i] != expect[i].x) {
        pass = false;
        break;
      }
    }
    checkResult("uint1-1D", pass);
    if (PRINT_PASS || !pass) {
      for (int i = 0; i < w; ++i)
        cout << "{" << output[i] << "}, ";
      cout << endl;
    }
    cudaFree(output);
    pass = true;
  }

  { // 1 element int 2D image.
    const int h = 3;
    const int w = 2;
    int1 expect[h * w] = {
        {1}, {2}, // 1
        {3}, {4}, // 2
        {5}, {6}, // 3
    };
    auto *input = getInput<int1>(expect, w, h, cudaCreateChannelDesc<int1>());
    int *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel1<int1><<<1, 1>>>(output, surf, w, h);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h; ++i) {
      if (output[i] != expect[i].x) {
        pass = false;
        break;
      }
    }
    checkResult("int1-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << (int)output[w * i + j] << "}, ";
        cout << endl;
      }
    cudaFree(output);
    pass = true;
  }

  { // 1 element short 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    short1 expect[d * h * w] = {
        {1},  {2}, // 1.1
        {3},  {4}, // 1.2
        {5},  {6}, // 1.3

        {7},  {8},  // 2.1
        {9},  {10}, // 2.2
        {11}, {12}, // 2.3
    };
    auto *input =
        getInput<short1>(expect, w, h, d, cudaCreateChannelDesc<short1>());
    short *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel1<short1><<<1, 1>>>(output, surf, w, h, d);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h * d; ++i) {
      if (output[i] != expect[i].x) {
        pass = false;
        break;
      }
    }
    checkResult("short1-3D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k)
            cout << "{" << output[w * h * i + j * w + k] << "}, ";
          cout << endl;
        }
        cout << endl;
      }
    cudaFree(output);
    pass = true;
  }

  { // 2 element char 1D image.
    const int w = 4;
    char2 expect[w] = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    };
    auto *input = getInput<char2>(expect, w, cudaCreateChannelDesc<char2>());
    char *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel2<char2><<<1, 1>>>(output, surf, w);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w; ++i) {
      if (output[2 * i] != expect[i].x || output[2 * i + 1] != expect[i].y) {
        pass = false;
        break;
      }
    }
    checkResult("char2-1D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < w; ++i)
        cout << "{" << (int)output[2 * i] << ", " << (int)output[2 * i + 1]
             << "}, ";
    cout << endl;
    cudaFree(output);
    pass = true;
  }

  { // 2 element uchar 2D image.
    const int h = 3;
    const int w = 2;
    uchar2 expect[h * w] = {
        {1, 2},  {3, 4},   // 1
        {5, 6},  {7, 8},   // 2
        {9, 10}, {11, 12}, // 3
    };
    auto *input =
        getInput<uchar2>(expect, w, h, cudaCreateChannelDesc<uchar2>());
    unsigned char *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel2<uchar2><<<1, 1>>>(output, surf, w, h);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h; ++i) {
      if (output[2 * i] != expect[i].x || output[2 * i + 1] != expect[i].y) {
        pass = false;
        break;
      }
    }
    checkResult("uchar2-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << (int)output[2 * (w * i + j)] << ", "
               << (int)output[2 * (w * i + j) + 1] << "}, ";
        cout << endl;
      }
    cudaFree(output);
    pass = true;
  }

  { // 2 element ushort 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    ushort2 expect[d * h * w] = {
        {1, 2},   {3, 4},   // 1.1
        {5, 6},   {7, 8},   // 1.2
        {9, 10},  {11, 12}, // 1.3

        {13, 14}, {15, 16}, // 2.1
        {17, 18}, {19, 20}, // 2.2
        {21, 22}, {23, 24}, // 2.3
    };
    auto *input =
        getInput<ushort2>(expect, w, h, d, cudaCreateChannelDesc<ushort2>());
    unsigned short *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel2<ushort2><<<1, 1>>>(output, surf, w, h, d);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h * d; ++i) {
      if (output[2 * i] != expect[i].x || output[2 * i + 1] != expect[i].y) {
        pass = false;
        break;
      }
    }
    checkResult("ushort2-3D", pass);
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
    cudaFree(output);
    pass = true;
  }

  { // 4 element float 1D image.
    const int w = 4;
    float4 expect[w] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    };
    auto *input = getInput<float4>(expect, w, cudaCreateChannelDesc<float4>());
    float *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel4<float4><<<1, 1>>>(output, surf, w);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    float precision = 0.001;
    for (int i = 0; i < w; ++i) {
      if ((output[4 * i] < expect[i].x - precision ||
           output[4 * i] > expect[i].x + precision) ||
          (output[4 * i + 1] < expect[i].y - precision ||
           output[4 * i + 1] > expect[i].y + precision) ||
          (output[4 * i + 2] < expect[i].z - precision ||
           output[4 * i + 2] > expect[i].z + precision) ||
          (output[4 * i + 3] < expect[i].w - precision ||
           output[4 * i + 3] > expect[i].w + precision)) {
        pass = false;
        break;
      }
    }
    checkResult("float4-1D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < w; ++i)
        cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
             << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
    cout << endl;
    cudaFree(output);
    pass = true;
  }

  { // 4 element int 2D image.
    const int h = 3;
    const int w = 2;
    int4 expect[h * w] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},     // 1
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 2
        {17, 18, 19, 20}, {21, 22, 23, 24}, // 3
    };
    auto *input = getInput<int4>(expect, w, h, cudaCreateChannelDesc<int4>());
    int *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel4<int4><<<1, 1>>>(output, surf, w, h);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h; ++i) {
      if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
          output[4 * i + 2] != expect[i].z ||
          output[4 * i + 3] != expect[i].w) {
        pass = false;
        break;
      }
    }
    checkResult("int4-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << output[4 * (w * i + j)] << ", "
               << output[4 * (w * i + j) + 1] << ", "
               << output[4 * (w * i + j) + 2] << ", "
               << output[4 * (w * i + j) + 3] << "}, ";
        cout << endl;
      }
    cudaFree(output);
    pass = true;
  }

  { // 4 element uint 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    uint4 expect[d * h * w] = {
        {1, 2, 3, 4},     {5, 6, 7, 8},     // 1.1
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.2
        {17, 18, 19, 20}, {21, 22, 23, 24}, // 1.3

        {25, 26, 27, 28}, {29, 30, 31, 32}, // 2.1
        {33, 34, 35, 36}, {37, 38, 39, 40}, // 2.2
        {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.3
    };
    auto *input =
        getInput<uint4>(expect, w, h, d, cudaCreateChannelDesc<uint4>());
    unsigned int *output;
    cudaMallocManaged(&output, sizeof(expect));
    auto surf = getSurf(input);
    kernel4<uint4><<<1, 1>>>(output, surf, w, h, d);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaFreeArray(input);
    for (int i = 0; i < w * h * d; ++i) {
      if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
          output[4 * i + 2] != expect[i].z ||
          output[4 * i + 3] != expect[i].w) {
        pass = false;
        break;
      }
    }
    checkResult("uint4-3D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k)
            cout << "{" << output[4 * (w * h * i + j * w + k)] << ", "
                 << output[4 * (w * h * i + j * w + k) + 1] << ", "
                 << output[4 * (w * h * i + j * w + k) + 2] << ", "
                 << output[4 * (w * h * i + j * w + k) + 3] << "}, ";
          cout << endl;
        }
        cout << endl;
      }
    cudaFree(output);
    pass = true;
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
