// ===-------- text_experimental_obj_driver_api.cu ------- *- CUDA -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include "cuda.h"
#include <iostream>
#include <vector>

#define PRINT_PASS 1

using namespace std;

int passed = 0;
int failed = 0;

const int d = 2;
const int h = 2;
const int w = 4;
const float2 data1d[w] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
const float2 data2d[h * w] = {
    {1, 2},  {3, 4},   {5, 6},   {7, 8},   // 1
    {9, 10}, {11, 12}, {13, 14}, {15, 16}, // 2
};
const float2 data3d[d * h * w] = {
    {1, 2},   {3, 4},   {5, 6},   {7, 8},   // 1:1
    {9, 10},  {11, 12}, {13, 14}, {15, 16}, // 1:2

    {17, 18}, {19, 20}, {21, 22}, {23, 24}, // 2:1
    {25, 26}, {27, 28}, {29, 30}, {31, 32}, // 2:2
};

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

void testCuMemcpy2D(const string &name, CUDA_MEMCPY2D p2d,
                    const vector<float2> &expect, bool isAsync) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, h * w * sizeof(float2));
  p2d.dstHost = output;
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpy2DAsync(&p2d, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpy2D(&p2d);
  }

  float precision = 0.001;
  for (int i = 0; i < w * h; ++i) {
    if ((output[2 * i] < expect[i].x - precision ||
         output[2 * i] > expect[i].x + precision) ||
        (output[2 * i + 1] < expect[i].y - precision ||
         output[2 * i + 1] > expect[i].y + precision)) {
      pass = false;
      break;
    }
  }
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j)
        cout << "{" << output[2 * (w * i + j)] << ", "
             << output[2 * (w * i + j) + 1] << "}, ";
      cout << endl;
    }
}

void testCuMemcpy2D(const CUarray &a) {
  CUDA_MEMCPY2D p2d = {0};
  p2d.srcArray = a;
  p2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  p2d.dstMemoryType = CU_MEMORYTYPE_HOST;

  vector<float2> expect = {
      {1, 2},  {3, 4},   {5, 6},   {7, 8},   // 1
      {9, 10}, {11, 12}, {13, 14}, {15, 16}, // 2
  };
  p2d.srcXInBytes = 0;
  p2d.srcY = 0;
  p2d.dstPitch = w * sizeof(float2);
  p2d.dstXInBytes = 0;
  p2d.dstY = 0;
  p2d.WidthInBytes = w * sizeof(float2);
  p2d.Height = h;
  testCuMemcpy2D("CuMemcpy2D", p2d, expect, false);
  testCuMemcpy2D("CuMemcpy2DAsync", p2d, expect, true);

  expect = {
      {11, 12}, {13, 14}, {15, 16}, {0, 0}, // 1
      {0, 0},   {0, 0},   {0, 0},   {0, 0}, // 2
  };
  p2d.srcXInBytes = 1 * sizeof(float2);
  p2d.srcY = 1;
  p2d.dstPitch = w * sizeof(float2);
  p2d.dstXInBytes = 0;
  p2d.dstY = 0;
  p2d.WidthInBytes = (w - 1) * sizeof(float2);
  p2d.Height = h - 1;
  testCuMemcpy2D("CuMemcpy2D:1,1,0,0", p2d, expect, false);
  testCuMemcpy2D("CuMemcpy2DAsync:1,1,0,0", p2d, expect, true);

  expect = {
      {0, 0}, {0, 0}, {0, 0}, {0, 0}, // 1
      {0, 0}, {1, 2}, {3, 4}, {5, 6}, // 2
  };
  p2d.srcXInBytes = 0;
  p2d.srcY = 0;
  p2d.dstPitch = w * sizeof(float2);
  p2d.dstXInBytes = 1 * sizeof(float2);
  p2d.dstY = 1;
  p2d.WidthInBytes = (w - 1) * sizeof(float2);
  p2d.Height = h - 1;
  testCuMemcpy2D("CuMemcpy2D:0,0,1,1", p2d, expect, false);
  testCuMemcpy2D("CuMemcpy2DAsync:0,0,1,1", p2d, expect, true);

  expect = {
      {0, 0}, {0, 0},   {0, 0},   {0, 0},   // 1
      {0, 0}, {11, 12}, {13, 14}, {15, 16}, // 2
  };
  p2d.srcXInBytes = 1 * sizeof(float2);
  p2d.srcY = 1;
  p2d.dstPitch = w * sizeof(float2);
  p2d.dstXInBytes = 1 * sizeof(float2);
  p2d.dstY = 1;
  p2d.WidthInBytes = (w - 1) * sizeof(float2);
  p2d.Height = h - 1;
  testCuMemcpy2D("CuMemcpy2D:1,1,1,1", p2d, expect, false);
  testCuMemcpy2D("CuMemcpy2DAsync:1,1,1,1", p2d, expect, true);
}

void testCuMemcpy3D(const string &name, CUDA_MEMCPY3D p3d,
                    const vector<float2> &expect, bool isAsync) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, h * w * d * sizeof(float2));
  p3d.dstHost = output;
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpy3DAsync(&p3d, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpy3D(&p3d);
  }

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
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int k = 0; k < d; ++k) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << output[2 * (w * (h * k + i) + j)] << ", "
               << output[2 * (w * (h * k + i) + j) + 1] << "}, ";
        cout << endl;
      }
      cout << endl;
    }
}

void testCuMemcpy3D(const CUarray &a) {
  CUDA_MEMCPY3D p3d = {0};
  p3d.srcArray = a;
  p3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  p3d.dstMemoryType = CU_MEMORYTYPE_HOST;

  vector<float2> expect = {
      {1, 2},   {3, 4},   {5, 6},   {7, 8},   // 1:1
      {9, 10},  {11, 12}, {13, 14}, {15, 16}, // 1:2

      {17, 18}, {19, 20}, {21, 22}, {23, 24}, // 2:1
      {25, 26}, {27, 28}, {29, 30}, {31, 32}, // 2:2
  };
  p3d.srcXInBytes = 0;
  p3d.srcY = 0;
  p3d.srcZ = 0;
  p3d.dstXInBytes = 0;
  p3d.dstY = 0;
  p3d.dstZ = 0;
  p3d.dstPitch = w * sizeof(float2);
  p3d.dstHeight = h;
  p3d.WidthInBytes = w * sizeof(float2);
  p3d.Height = h;
  p3d.Depth = d;
  testCuMemcpy3D("CuMemcpy3D", p3d, expect, false);
  testCuMemcpy3D("CuMemcpy3DAsync", p3d, expect, true);

  expect = {
      {27, 28}, {29, 30}, {31, 32}, {0, 0}, // 1:1
      {0, 0},   {0, 0},   {0, 0},   {0, 0}, // 1:2

      {0, 0},   {0, 0},   {0, 0},   {0, 0}, // 2:1
      {0, 0},   {0, 0},   {0, 0},   {0, 0}, // 2:2
  };
  p3d.srcXInBytes = 1 * sizeof(float2);
  p3d.srcY = 1;
  p3d.srcZ = 1;
  p3d.dstXInBytes = 0;
  p3d.dstY = 0;
  p3d.dstZ = 0;
  p3d.dstPitch = w * sizeof(float2);
  p3d.dstHeight = h;
  p3d.WidthInBytes = (w - 1) * sizeof(float2);
  p3d.Height = h - 1;
  p3d.Depth = d - 1;
  testCuMemcpy3D("CuMemcpy3D:1,1,1,0,0,0", p3d, expect, false);
  testCuMemcpy3D("CuMemcpy3DAsync:1,1,1,0,0,0", p3d, expect, true);

  expect = {
      {0, 0}, {0, 0}, {0, 0}, {0, 0}, // 1:1
      {0, 0}, {0, 0}, {0, 0}, {0, 0}, // 1:2

      {0, 0}, {0, 0}, {0, 0}, {0, 0}, // 2:1
      {0, 0}, {1, 2}, {3, 4}, {5, 6}, // 2:2
  };
  p3d.srcXInBytes = 0;
  p3d.srcY = 0;
  p3d.srcZ = 0;
  p3d.dstXInBytes = 1 * sizeof(float2);
  p3d.dstY = 1;
  p3d.dstZ = 1;
  p3d.dstPitch = w * sizeof(float2);
  p3d.dstHeight = h;
  p3d.WidthInBytes = (w - 1) * sizeof(float2);
  p3d.Height = h - 1;
  p3d.Depth = d - 1;
  testCuMemcpy3D("CuMemcpy3D:0,0,0,1,1,1", p3d, expect, false);
  testCuMemcpy3D("CuMemcpy3DAsync:0,0,0,1,1,1", p3d, expect, true);

  expect = {
      {0, 0}, {0, 0},   {0, 0},   {0, 0}, // 1:1
      {0, 0}, {0, 0},   {0, 0},   {0, 0}, // 1:2

      {0, 0}, {0, 0},   {0, 0},   {0, 0},   // 2:1
      {0, 0}, {27, 28}, {29, 30}, {31, 32}, // 2:2
  };
  p3d.srcXInBytes = 1 * sizeof(float2);
  p3d.srcY = 1;
  p3d.srcZ = 1;
  p3d.dstXInBytes = 1 * sizeof(float2);
  p3d.dstY = 1;
  p3d.dstZ = 1;
  p3d.dstPitch = w * sizeof(float2);
  p3d.dstHeight = h;
  p3d.WidthInBytes = (w - 1) * sizeof(float2);
  p3d.Height = h - 1;
  p3d.Depth = d - 1;
  testCuMemcpy3D("CuMemcpy3D:1,1,1,1,1,1", p3d, expect, false);
  testCuMemcpy3D("CuMemcpy3DAsync:1,1,1,1,1,1", p3d, expect, true);
}

void checkArray(const string &name, const CUarray &a,
                const vector<float2> &expect) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, w * sizeof(float2));
  cuMemcpyAtoH(output, a, 0, w * sizeof(float2));
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
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < w; ++i)
      cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
  cout << endl;
}

void testCuMemcpyAtoA(const string &name, const CUarray &a, size_t dstOffset,
                      size_t srctOffset, size_t size,
                      const vector<float2> &expect) {
  CUarray output;
  CUDA_ARRAY_DESCRIPTOR d1d;
  d1d.Width = w;
  d1d.Height = 0;
  d1d.Format = CU_AD_FORMAT_FLOAT;
  d1d.NumChannels = 2;
  cuArrayCreate(&output, &d1d);
  cuMemcpyAtoA(output, dstOffset, a, srctOffset, size);
  checkArray(name, output, expect);
}

void testCuMemcpyAtoA(const CUarray &a) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyAtoA("testCuMemcpyAtoA", a, 0, 0, w * sizeof(float2), expect);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyAtoA("testCuMemcpyAtoA:0,0;-1", a, 0, 0, (w - 1) * sizeof(float2),
                   expect);
  expect = {{0, 0}, {1, 2}, {3, 4}, {5, 6}};
  testCuMemcpyAtoA("testCuMemcpyAtoA:1,0;-1", a, 1 * sizeof(float2), 0,
                   (w - 1) * sizeof(float2), expect);
  expect = {{3, 4}, {5, 6}, {7, 8}, {0, 0}};
  testCuMemcpyAtoA("testCuMemcpyAtoA:0,1;-1", a, 0, 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect);
  expect = {{0, 0}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyAtoA("testCuMemcpyAtoA:1,1;-1", a, 1 * sizeof(float2),
                   1 * sizeof(float2), (w - 1) * sizeof(float2), expect);
}

void checkDevicePtr(const string &name, const CUdeviceptr &ptr,
                    const vector<float2> &expect) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, w * sizeof(float2));
  cuMemcpyDtoH(output, ptr, w * sizeof(float2));
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
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < w; ++i)
      cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
  cout << endl;
}

void testCuMemcpyAtoD(const string &name, const CUarray &a, size_t srctOffset,
                      size_t size, const vector<float2> &expect) {
  CUdeviceptr output;
  cuMemAlloc(&output, w * sizeof(float2));
  cuMemcpyAtoD(output, a, srctOffset, size);
  checkDevicePtr(name, output, expect);
}

void testCuMemcpyAtoD(const CUarray &a) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyAtoD("testCuMemcpyAtoD", a, 0, w * sizeof(float2), expect);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyAtoD("testCuMemcpyAtoD:0;-1", a, 0, (w - 1) * sizeof(float2),
                   expect);
  expect = {{3, 4}, {5, 6}, {7, 8}, {0, 0}};
  testCuMemcpyAtoD("testCuMemcpyAtoD:1;-1", a, 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect);
}

void testCuMemcpyAtoH(const string &name, const CUarray &a, size_t srctOffset,
                      size_t size, const vector<float2> &expect, bool isAsync) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, w * sizeof(float2));
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpyAtoHAsync(output, a, srctOffset, size, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpyAtoH(output, a, srctOffset, size);
  }
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
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < w; ++i)
      cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
  cout << endl;
}

void testCuMemcpyAtoH(const CUarray &a) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyAtoH("testCuMemcpyAtoH", a, 0, w * sizeof(float2), expect, false);
  testCuMemcpyAtoH("testCuMemcpyAtoHAsync", a, 0, w * sizeof(float2), expect,
                   true);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyAtoH("testCuMemcpyAtoH:0;-1", a, 0, (w - 1) * sizeof(float2),
                   expect, false);
  testCuMemcpyAtoH("testCuMemcpyAtoHAsync:0;-1", a, 0, (w - 1) * sizeof(float2),
                   expect, true);
  expect = {{3, 4}, {5, 6}, {7, 8}, {0, 0}};
  testCuMemcpyAtoH("testCuMemcpyAtoH:1;-1", a, 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect, false);
  testCuMemcpyAtoH("testCuMemcpyAtoHAsync:1;-1", a, 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect, true);
}

void testCuMemcpyDtoA(const string &name, const CUdeviceptr &d,
                      size_t dstOffset, size_t size,
                      const vector<float2> &expect) {
  CUarray output;
  CUDA_ARRAY_DESCRIPTOR d1d;
  d1d.Width = w;
  d1d.Height = 0;
  d1d.Format = CU_AD_FORMAT_FLOAT;
  d1d.NumChannels = 2;
  cuArrayCreate(&output, &d1d);
  cuMemcpyDtoA(output, dstOffset, d, size);
  checkArray(name, output, expect);
}

void testCuMemcpyDtoA(const CUdeviceptr &d) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyDtoA("testCuMemcpyDtoA", d, 0, w * sizeof(float2), expect);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyDtoA("testCuMemcpyDtoA:0;-1", d, 0, (w - 1) * sizeof(float2),
                   expect);
  expect = {{0, 0}, {1, 2}, {3, 4}, {5, 6}};
  testCuMemcpyDtoA("testCuMemcpyDtoA:1;-1", d, 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect);
}

void testCuMemcpyDtoD(const string &name, const CUdeviceptr &d, size_t size,
                      const vector<float2> &expect, bool isAsync) {
  CUdeviceptr output;
  cuMemAlloc(&output, w * sizeof(float2));
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpyDtoDAsync(output, d, size, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpyDtoD(output, d, size);
  }
  checkDevicePtr(name, output, expect);
}

void testCuMemcpyDtoD(const CUdeviceptr &d) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyDtoD("testCuMemcpyDtoD", d, w * sizeof(float2), expect, false);
  testCuMemcpyDtoD("testCuMemcpyDtoDAsync", d, w * sizeof(float2), expect,
                   true);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyDtoD("testCuMemcpyDtoD:-1", d, (w - 1) * sizeof(float2), expect,
                   false);
  testCuMemcpyDtoD("testCuMemcpyDtoDAsync:-1", d, (w - 1) * sizeof(float2),
                   expect, true);
}

void testCuMemcpyDtoH(const string &name, const CUdeviceptr &d, size_t size,
                      const vector<float2> &expect, bool isAsync) {
  bool pass = true;
  float *output;
  cudaMallocManaged(&output, w * sizeof(float2));
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpyDtoHAsync(output, d, size, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpyDtoH(output, d, size);
  }
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
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < w; ++i)
      cout << "{" << output[2 * i] << ", " << output[2 * i + 1] << "}, ";
  cout << endl;
}

void testCuMemcpyDtoH(const CUdeviceptr &d) {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyDtoH("testCuMemcpyDtoH", d, w * sizeof(float2), expect, false);
  testCuMemcpyDtoH("testCuMemcpyDtoHAsync", d, w * sizeof(float2), expect,
                   true);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyDtoH("testCuMemcpyDtoH:-1", d, (w - 1) * sizeof(float2), expect,
                   false);
  testCuMemcpyDtoH("testCuMemcpyDtoHAsync:-1", d, (w - 1) * sizeof(float2),
                   expect, true);
}

void testCuMemcpyHtoA(const string &name, size_t dstOffset, size_t size,
                      const vector<float2> &expect, bool isAsync) {
  CUarray output;
  CUDA_ARRAY_DESCRIPTOR d1d;
  d1d.Width = w;
  d1d.Height = 0;
  d1d.Format = CU_AD_FORMAT_FLOAT;
  d1d.NumChannels = 2;
  cuArrayCreate(&output, &d1d);
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpyHtoAAsync(output, dstOffset, data1d, size, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpyHtoA(output, dstOffset, data1d, size);
  }
  checkArray(name, output, expect);
}

void testCuMemcpyHtoA() {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyHtoA("testCuMemcpyHtoA", 0, w * sizeof(float2), expect, false);
  testCuMemcpyHtoA("testCuMemcpyHtoAAsync", 0, w * sizeof(float2), expect,
                   true);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyHtoA("testCuMemcpyHtoA:0;-1", 0, (w - 1) * sizeof(float2), expect,
                   false);
  testCuMemcpyHtoA("testCuMemcpyHtoAAsync:0;-1", 0, (w - 1) * sizeof(float2),
                   expect, true);
  expect = {{0, 0}, {1, 2}, {3, 4}, {5, 6}};
  testCuMemcpyHtoA("testCuMemcpyHtoA:1;-1", 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect, false);
  testCuMemcpyHtoA("testCuMemcpyHtoAAsync:1;-1", 1 * sizeof(float2),
                   (w - 1) * sizeof(float2), expect, true);
}

void testCuMemcpyHtoD(const string &name, size_t size,
                      const vector<float2> &expect, bool isAsync) {
  CUdeviceptr output;
  cuMemAlloc(&output, w * sizeof(float2));
  if (isAsync) {
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    cuMemcpyHtoDAsync(output, data1d, size, s);
    cuStreamSynchronize(s);
  } else {
    cuMemcpyHtoD(output, data1d, size);
  }
  checkDevicePtr(name, output, expect);
}

void testCuMemcpyHtoD() {
  vector<float2> expect = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  testCuMemcpyHtoD("testCuMemcpyHtoD", w * sizeof(float2), expect, false);
  testCuMemcpyHtoD("testCuMemcpyHtoDAsync", w * sizeof(float2), expect, true);
  expect = {{1, 2}, {3, 4}, {5, 6}, {0, 0}};
  testCuMemcpyHtoD("testCuMemcpyHtoD:-1", (w - 1) * sizeof(float2), expect,
                   false);
  testCuMemcpyHtoD("testCuMemcpyHtoDAsync:-1", (w - 1) * sizeof(float2), expect,
                   true);
}

int main() {
  cuInit(0);
  CUdevice dev = 0;
  cuDeviceGet(&dev, 0);
  CUcontext ctx = 0;
  cuCtxCreate(&ctx, 0, dev);
  CUarray a;

  CUDA_ARRAY_DESCRIPTOR d2d;
  d2d.Width = w;
  d2d.Height = h;
  d2d.Format = CU_AD_FORMAT_FLOAT;
  d2d.NumChannels = 2;
  cuArrayCreate(&a, &d2d);
  CUDA_MEMCPY2D p2d = {0};
  p2d.srcMemoryType = CU_MEMORYTYPE_HOST;
  p2d.srcPitch = w * sizeof(float2);
  p2d.srcHost = data2d;
  p2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  p2d.dstArray = a;
  p2d.WidthInBytes = w * sizeof(float2);
  p2d.Height = h;
  cuMemcpy2D(&p2d);

  testCuMemcpy2D(a);

  cuArrayDestroy(a);

  CUDA_ARRAY3D_DESCRIPTOR d3d;
  d3d.Width = w;
  d3d.Height = h;
  d3d.Depth = d;
  d3d.Format = CU_AD_FORMAT_FLOAT;
  d3d.NumChannels = 2;
  cuArray3DCreate(&a, &d3d);
  CUDA_MEMCPY3D p3d = {0};
  p3d.srcMemoryType = CU_MEMORYTYPE_HOST;
  p3d.srcPitch = w * sizeof(float2);
  p3d.srcHost = data3d;
  p3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  p3d.dstArray = a;
  p3d.WidthInBytes = w * sizeof(float2);
  p3d.Height = h;
  p3d.Depth = d;
  cuMemcpy3D(&p3d);

  testCuMemcpy3D(a);

  cuArrayDestroy(a);

  CUDA_ARRAY_DESCRIPTOR d1d;
  d1d.Width = w;
  d1d.Height = 0;
  d1d.Format = CU_AD_FORMAT_FLOAT;
  d1d.NumChannels = 2;
  cuArrayCreate(&a, &d1d);
  cuMemcpyHtoA(a, 0, data1d, w * sizeof(float2));
  CUdeviceptr d;
  cuMemAlloc(&d, w * sizeof(float2));
  cuMemcpyHtoD(d, data1d, w * sizeof(float2));

  testCuMemcpyAtoA(a);
  testCuMemcpyAtoD(a);
  testCuMemcpyAtoH(a);
  testCuMemcpyDtoA(d);
  testCuMemcpyDtoD(d);
  testCuMemcpyDtoH(d);
  testCuMemcpyHtoA();
  testCuMemcpyHtoD();

  cuArrayDestroy(a);

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
