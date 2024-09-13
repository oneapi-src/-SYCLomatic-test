// ===-------- text_experimental_obj_memcpy2d_api.cu ----- *- CUDA -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>
#include <vector>

#define PRINT_PASS 1

using namespace std;

int passed = 0;
int failed = 0;

const int h = 3;
const int w = 3;
const short2 input[h * w] = {
    {1, 2},   {3, 4},   {5, 6},   // 1
    {7, 8},   {9, 10},  {11, 12}, // 2
    {13, 14}, {15, 16}, {17, 18}, // 3
};
const auto desc = cudaCreateChannelDesc<short2>();

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

void checkArray(const string &name, const cudaArray_t &a,
                const vector<short2> &expect) {
  bool pass = true;
  short *output;
  cudaMallocManaged(&output, w * h * sizeof(short2));
  cudaMemcpy2DFromArray(output, w * sizeof(short2), a, 0, 0, w * sizeof(short2),
                        h, cudaMemcpyDeviceToHost);
  for (int i = 0; i < w; ++i) {
    if (output[2 * i] != expect[i].x || output[2 * i + 1] != expect[i].y) {
      pass = false;
      break;
    }
  }
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const auto id = i * w + j;
        cout << "{" << output[2 * id] << ", " << output[2 * id + 1] << "}, ";
      }
      cout << endl;
    }
  cout << endl;
}

void testCudaMemcpy2DArrayToArray(const string &name, const cudaArray_t &a,
                                  size_t w_offest_src, size_t h_offest_src,
                                  size_t w_offest_dest, size_t h_offest_dest,
                                  size_t w_size, size_t h_size,
                                  const vector<short2> &expect) {
  cudaArray_t output;
  cudaMallocArray(&output, &desc, w, h);
  cudaMemcpy2DArrayToArray(output, w_offest_dest, h_offest_dest, a,
                           w_offest_src, h_offest_src, w_size, h_size,
                           cudaMemcpyDeviceToDevice);
  checkArray(name, output, expect);
}

void testCudaMemcpy2DArrayToArray(const cudaArray_t &a) {
  vector<short2> expect = {
      {1, 2},   {3, 4},   {5, 6},   // 1
      {7, 8},   {9, 10},  {11, 12}, // 2
      {13, 14}, {15, 16}, {17, 18}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray", a, 0, 0, 0, 0,
                               w * sizeof(short2), h, expect);
  expect = {
      {1, 2},   {3, 4},   {0, 0}, // 1
      {7, 8},   {9, 10},  {0, 0}, // 2
      {13, 14}, {15, 16}, {0, 0}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:0,0;0,0;-1,0", a, 0, 0,
                               0, 0, (w - 1) * sizeof(short2), h, expect);
  expect = {
      {1, 2}, {3, 4},  {5, 6},   // 1
      {7, 8}, {9, 10}, {11, 12}, // 2
      {0, 0}, {0, 0},  {0, 0},   // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:0,0;0,0;0,-1", a, 0, 0,
                               0, 0, w * sizeof(short2), h - 1, expect);
  expect = {
      {3, 4},   {5, 6},   {0, 0}, // 1
      {9, 10},  {11, 12}, {0, 0}, // 2
      {15, 16}, {17, 18}, {0, 0}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:1,0;0,0;-1,0", a,
                               1 * sizeof(short2), 0, 0, 0,
                               (w - 1) * sizeof(short2), h, expect);
  expect = {
      {7, 8},   {9, 10},  {11, 12}, // 1
      {13, 14}, {15, 16}, {17, 18}, // 2
      {0, 0},   {0, 0},   {0, 0},   // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:0,1;0,0;0,-1", a, 0, 1,
                               0, 0, w * sizeof(short2), h - 1, expect);
  expect = {
      {0, 0}, {1, 2},   {3, 4},   // 1
      {0, 0}, {7, 8},   {9, 10},  // 2
      {0, 0}, {13, 14}, {15, 16}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:0,0;1,0;-1,0", a, 0, 0,
                               1 * sizeof(short2), 0, (w - 1) * sizeof(short2),
                               h, expect);
  expect = {
      {0, 0}, {0, 0},  {0, 0},   // 1
      {1, 2}, {3, 4},  {5, 6},   // 2
      {7, 8}, {9, 10}, {11, 12}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:0,0;0,1;0,-1", a, 0, 0,
                               0, 1, w * sizeof(short2), h - 1, expect);
  expect = {
      {0, 0}, {0, 0},   {0, 0},   // 1
      {0, 0}, {9, 10},  {11, 12}, // 2
      {0, 0}, {15, 16}, {17, 18}, // 3
  };
  testCudaMemcpy2DArrayToArray("cudaMemcpy2DArrayToArray:1,1;1,1;-1,-1", a,
                               1 * sizeof(short2), 1, 1 * sizeof(short2), 1,
                               (w - 1) * sizeof(short2), h - 1, expect);
}

void testCudaMemcpy2DFromArray(const string &name, const cudaArray_t &a,
                               size_t w_offest_src, size_t h_offest_src,
                               size_t pitch_dest, size_t w_size, size_t h_size,
                               const vector<short2> &expect, bool isAsync) {
  bool pass = true;
  short *output;
  cudaMallocManaged(&output, w * h * sizeof(short2));
  if (isAsync) {
    cudaMemcpy2DFromArrayAsync(output, pitch_dest, a, w_offest_src,
                               h_offest_src, w_size, h_size,
                               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy2DFromArray(output, pitch_dest, a, w_offest_src, h_offest_src,
                          w_size, h_size, cudaMemcpyDeviceToHost);
  }
  for (int i = 0; i < w; ++i) {
    if (output[2 * i] != expect[i].x || output[2 * i + 1] != expect[i].y) {
      pass = false;
      break;
    }
  }
  checkResult(name, pass);
  if (PRINT_PASS || !pass)
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const auto id = i * w + j;
        cout << "{" << output[2 * id] << ", " << output[2 * id + 1] << "}, ";
      }
      cout << endl;
    }
  cout << endl;
}

void testCudaMemcpy2DFromArray(const cudaArray_t &a) {
  vector<short2> expect = {
      {1, 2},   {3, 4},   {5, 6},   // 1
      {7, 8},   {9, 10},  {11, 12}, // 2
      {13, 14}, {15, 16}, {17, 18}, // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray", a, 0, 0,
                            w * sizeof(short2), w * sizeof(short2), h, expect,
                            false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync", a, 0, 0,
                            w * sizeof(short2), w * sizeof(short2), h, expect,
                            true);
  expect = {
      {1, 2},   {3, 4},   {0, 0}, // 1
      {7, 8},   {9, 10},  {0, 0}, // 2
      {13, 14}, {15, 16}, {0, 0}, // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:0,0;0;-1,0", a, 0, 0,
                            w * sizeof(short2), (w - 1) * sizeof(short2), h,
                            expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:0,0;0;-1,0", a, 0, 0,
                            w * sizeof(short2), (w - 1) * sizeof(short2), h,
                            expect, true);
  expect = {
      {1, 2}, {3, 4},  {5, 6},   // 1
      {7, 8}, {9, 10}, {11, 12}, // 2
      {0, 0}, {0, 0},  {0, 0},   // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:0,0;0;0,-1", a, 0, 0,
                            w * sizeof(short2), w * sizeof(short2), h - 1,
                            expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:0,0;0;0,-1", a, 0, 0,
                            w * sizeof(short2), w * sizeof(short2), h - 1,
                            expect, true);
  expect = {
      {3, 4},   {5, 6},   {0, 0}, // 1
      {9, 10},  {11, 12}, {0, 0}, // 2
      {15, 16}, {17, 18}, {0, 0}, // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:1,0;0;-1,0", a,
                            1 * sizeof(short2), 0, w * sizeof(short2),
                            (w - 1) * sizeof(short2), h, expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:1,0;0;-1,0", a,
                            1 * sizeof(short2), 0, w * sizeof(short2),
                            (w - 1) * sizeof(short2), h, expect, true);
  expect = {
      {7, 8},   {9, 10},  {11, 12}, // 1
      {13, 14}, {15, 16}, {17, 18}, // 2
      {0, 0},   {0, 0},   {0, 0},   // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:0,1;0;0,-1", a, 0, 1,
                            w * sizeof(short2), w * sizeof(short2), h - 1,
                            expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:0,1;0;0,-1", a, 0, 1,
                            w * sizeof(short2), w * sizeof(short2), h - 1,
                            expect, true);
  expect = {
      {1, 2},  {3, 4},   {7, 8},   // 1
      {9, 10}, {13, 14}, {15, 16}, // 2
      {0, 0},  {0, 0},   {0, 0},   // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:0,0;-1;-1,0", a, 0, 0,
                            (w - 1) * sizeof(short2), (w - 1) * sizeof(short2),
                            h, expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:0,0;-1;-1,0", a, 0, 0,
                            (w - 1) * sizeof(short2), (w - 1) * sizeof(short2),
                            h, expect, true);
  expect = {
      {9, 10},  {11, 12}, {15, 16}, // 1
      {17, 18}, {0, 0},   {0, 0},   // 2
      {0, 0},   {0, 0},   {0, 0},   // 3
  };
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArray:1,1;-1;-1,1", a,
                            1 * sizeof(short2), 1, (w - 1) * sizeof(short2),
                            (w - 1) * sizeof(short2), h - 1, expect, false);
  testCudaMemcpy2DFromArray("cudaMemcpy2DFromArrayAsync:1,1;-1;-1,1", a,
                            1 * sizeof(short2), 1, (w - 1) * sizeof(short2),
                            (w - 1) * sizeof(short2), h - 1, expect, true);
}

void testCudaMemcpy2DToArray(const string &name, size_t pitch_src,
                             size_t w_offest_dest, size_t h_offest_dest,
                             size_t w_size, size_t h_size,
                             const vector<short2> &expect, bool isAsync) {
  cudaArray_t output;
  cudaMallocArray(&output, &desc, w, h);
  if (isAsync) {
    cudaMemcpy2DToArrayAsync(output, w_offest_dest, h_offest_dest, input,
                             pitch_src, w_size, h_size, cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy2DToArray(output, w_offest_dest, h_offest_dest, input, pitch_src,
                        w_size, h_size, cudaMemcpyHostToDevice);
  }
  checkArray(name, output, expect);
}

void testCudaMemcpy2DToArray() {
  vector<short2> expect = {
      {1, 2},   {3, 4},   {5, 6},   // 1
      {7, 8},   {9, 10},  {11, 12}, // 2
      {13, 14}, {15, 16}, {17, 18}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray", w * sizeof(short2), 0, 0,
                          w * sizeof(short2), h, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync", w * sizeof(short2), 0, 0,
                          w * sizeof(short2), h, expect, true);
  expect = {
      {1, 2},   {3, 4},   {0, 0}, // 1
      {7, 8},   {9, 10},  {0, 0}, // 2
      {13, 14}, {15, 16}, {0, 0}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:0;0,0;-1,0", w * sizeof(short2),
                          0, 0, (w - 1) * sizeof(short2), h, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:0;0,0;-1,0",
                          w * sizeof(short2), 0, 0, (w - 1) * sizeof(short2), h,
                          expect, true);
  expect = {
      {1, 2}, {3, 4},  {5, 6},   // 1
      {7, 8}, {9, 10}, {11, 12}, // 2
      {0, 0}, {0, 0},  {0, 0},   // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:0;0,0;0,-1", w * sizeof(short2),
                          0, 0, w * sizeof(short2), h - 1, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:0;0,0;0,-1",
                          w * sizeof(short2), 0, 0, w * sizeof(short2), h - 1,
                          expect, true);
  expect = {
      {0, 0}, {1, 2},   {3, 4},   // 1
      {0, 0}, {7, 8},   {9, 10},  // 2
      {0, 0}, {13, 14}, {15, 16}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:0;1,0;-1,0", w * sizeof(short2),
                          1 * sizeof(short2), 0, (w - 1) * sizeof(short2), h,
                          expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:0;1,0;-1,0",
                          w * sizeof(short2), 1 * sizeof(short2), 0,
                          (w - 1) * sizeof(short2), h, expect, true);
  expect = {
      {0, 0}, {0, 0},  {0, 0},   // 1
      {1, 2}, {3, 4},  {5, 6},   // 2
      {7, 8}, {9, 10}, {11, 12}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:0;0,1;0,-1", w * sizeof(short2),
                          0, 1, w * sizeof(short2), h - 1, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:0;0,1;0,-1",
                          w * sizeof(short2), 0, 1, w * sizeof(short2), h - 1,
                          expect, true);
  expect = {
      {1, 2},   {3, 4},   {0, 0}, // 1
      {7, 8},   {9, 10},  {0, 0}, // 2
      {13, 14}, {15, 16}, {0, 0}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:-1;0,0;-1,0",
                          (w - 1) * sizeof(short2), 0, 0,
                          (w - 1) * sizeof(short2), h, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:-1;0,0;-1,0",
                          (w - 1) * sizeof(short2), 0, 0,
                          (w - 1) * sizeof(short2), h, expect, true);
  expect = {
      {0, 0}, {0, 0}, {0, 0}, // 1
      {0, 0}, {1, 2}, {3, 4}, // 2
      {0, 0}, {5, 6}, {7, 8}, // 3
  };
  testCudaMemcpy2DToArray("cudaMemcpy2DToArray:-1;1,1;-1,1",
                          (w - 1) * sizeof(short2), 1 * sizeof(short2), 1,
                          (w - 1) * sizeof(short2), h - 1, expect, false);
  testCudaMemcpy2DToArray("cudaMemcpy2DToArrayAsync:-1;1,1;-1,1",
                          (w - 1) * sizeof(short2), 1 * sizeof(short2), 1,
                          (w - 1) * sizeof(short2), h - 1, expect, true);
}

int main() {
  cudaArray_t a;
  cudaMallocArray(&a, &desc, w, h);
  cudaMemcpy2DToArray(a, 0, 0, input, sizeof(short2) * w, sizeof(short2) * w, h,
                      cudaMemcpyHostToDevice);

  testCudaMemcpy2DArrayToArray(a);
  testCudaMemcpy2DFromArray(a);
  testCudaMemcpy2DToArray();

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
