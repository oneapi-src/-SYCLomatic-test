// ===-------- text_experimental_obj_peer_api.cu ------- *- CUDA -* -------===//
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

__global__ void kernel(short *output, cudaTextureObject_t tex, int w, int h,
                       int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = tex3D<short4>(tex, k, j, i);
        output[4 * (w * h * i + w * j + k)] = ret.x;
        output[4 * (w * h * i + w * j + k) + 1] = ret.y;
        output[4 * (w * h * i + w * j + k) + 2] = ret.z;
        output[4 * (w * h * i + w * j + k) + 3] = ret.w;
      }
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

int main() {
  bool pass = true;

  const int d = 2;
  const int h = 2;
  const int w = 4;
  short4 input[d * h * w] = {
      {1, 2, 3, 4},     {5, 6, 7, 8},
      {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
      {17, 18, 19, 20}, {21, 22, 23, 24},
      {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
      {33, 34, 35, 36}, {37, 38, 39, 40},
      {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
      {49, 50, 51, 52}, {53, 54, 55, 56},
      {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
  };
  auto desc = cudaCreateChannelDesc<short4>();

  { // peer
    short *src;
    cudaMallocManaged(&src, sizeof(input));
    cudaMemcpy(src, input, sizeof(input), cudaMemcpyHostToDevice);
    {
      short4 expect[w] = {
          {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {13, 14, 15, 16},
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpyPeer(output, 0, src, 0, 4 * sizeof(short4));
      for (int i = 0; i < w; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpyPeer:4", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
               << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      short4 expect[w] = {
          {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {0, 0, 0, 0},
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpyPeer(output, 0, src, 0, 3 * sizeof(short4));
      for (int i = 0; i < w; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpyPeer:3", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
               << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      short4 expect[w] = {
          {1, 2, 3, 4},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpyPeerAsync(output, 0, src, 0, sizeof(short4));
      cudaDeviceSynchronize();
      for (int i = 0; i < w; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpyPeerAsync:1", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
               << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
      cout << endl;
      pass = true;
    }
    {
      short4 expect[w] = {
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
          {0, 0, 0, 0},
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpyPeerAsync(output, 0, src, 0, 0);
      cudaDeviceSynchronize();
      for (int i = 0; i < w; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpyPeerAsync:0", pass);
      if (PRINT_PASS || !pass)
        for (int i = 0; i < w; ++i)
          cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
               << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
      cout << endl;
      pass = true;
    }
  }

  { // p2p
    short *srcDevice;
    cudaMallocManaged(&srcDevice, sizeof(input));
    cudaMemcpy(srcDevice, input, sizeof(input), cudaMemcpyHostToDevice);
    const auto src = make_cudaPitchedPtr(srcDevice, w * sizeof(short4), w, h);
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4},     {5, 6, 7, 8},
          {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w * sizeof(short4), h, d);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2p:1", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent((w - 1) * sizeof(short4), h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2p:2", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcPos = make_cudaPos(1 * sizeof(short4), 1, 1);
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent((w - 1) * sizeof(short4), h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:p2p:3", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 2.1
          {0, 0, 0, 0}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstPos = make_cudaPos(1 * sizeof(short4), 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent((w - 1) * sizeof(short4), h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:p2p:4", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcPos = make_cudaPos(2 * sizeof(short4), 1, 1);
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstPos = make_cudaPos(2 * sizeof(short4), 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent((w - 2) * sizeof(short4), h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2p:5", pass);
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
      pass = true;
    }
  }

  { // p2a
    short *srcDevice;
    cudaMallocManaged(&srcDevice, sizeof(input));
    cudaMemcpy(srcDevice, input, sizeof(input), cudaMemcpyHostToDevice);
    const auto src = make_cudaPitchedPtr(srcDevice, w * sizeof(short4), w, h);
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4},     {5, 6, 7, 8},
          {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w, h, d);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2a:1", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2a:2", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcPos = make_cudaPos(1, 1, 1);
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:p2a:3", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 2.1
          {0, 0, 0, 0}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstPos = make_cudaPos(1, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:p2a:4", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcPtr = src;
      pp.srcPos = make_cudaPos(2, 1, 1);
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstPos = make_cudaPos(2, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 2, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:p2a:5", pass);
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
      pass = true;
    }
  }

  { // a2p
    cudaArray *src;
    cudaMalloc3DArray(&src, &desc, {w, h, d});
    cudaMemcpy3DParms p = {0};
    p.srcPtr = make_cudaPitchedPtr(input, w * sizeof(short4), w, h);
    p.dstArray = src;
    p.extent = make_cudaExtent(w, h, d);
    p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4},     {5, 6, 7, 8},
          {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w, h, d);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2p:1", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2p:2", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {0, 0, 0, 0}, // 1.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 1.2
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.1
          {0, 0, 0, 0},     {0, 0, 0, 0},
          {0, 0, 0, 0},     {0, 0, 0, 0}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcPos = make_cudaPos(1, 1, 1);
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:a2p:3", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},    // 2.1
          {0, 0, 0, 0}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstPos = make_cudaPos(1, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:a2p:4", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 1.2
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},     {0, 0, 0, 0},     // 2.1
          {0, 0, 0, 0}, {0, 0, 0, 0}, {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcPos = make_cudaPos(2, 1, 1);
      pp.srcDevice = 0;
      pp.dstPtr = make_cudaPitchedPtr(output, w * sizeof(short4), w, h);
      pp.dstPos = make_cudaPos(2, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 2, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2p:5", pass);
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
      pass = true;
    }
    cudaFreeArray(src);
  }

  { // a2a
    cudaArray *src;
    cudaMalloc3DArray(&src, &desc, {w, h, d});
    cudaMemcpy3DParms p = {0};
    p.srcPtr = make_cudaPitchedPtr(input, w * sizeof(short4), w, h);
    p.dstArray = src;
    p.extent = make_cudaExtent(w, h, d);
    p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4},     {5, 6, 7, 8},
          {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w, h, d);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2a:1", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {1, 2, 3, 4},     {5, 6, 7, 8},
          {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2a:2", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {53, 54, 55, 56},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcPos = make_cudaPos(1, 1, 1);
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:a2a:3", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {1, 2, 3, 4},
          {5, 6, 7, 8},     {9, 10, 11, 12}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstPos = make_cudaPos(1, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 1, h - 1, d - 1);
      cudaMemcpy3DPeerAsync(&pp);
      cudaDeviceSynchronize();
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeerAsync:a2a:4", pass);
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
      pass = true;
    }
    {
      short4 expect[d * h * w] = {
          {53, 54, 55, 56}, {57, 58, 59, 60},
          {61, 62, 63, 64}, {13, 14, 15, 16}, // 1.1
          {17, 18, 19, 20}, {21, 22, 23, 24},
          {25, 26, 27, 28}, {29, 30, 31, 32}, // 1.2
          {33, 34, 35, 36}, {37, 38, 39, 40},
          {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.1
          {49, 50, 51, 52}, {1, 2, 3, 4},
          {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.2
      };
      short *output;
      cudaMallocManaged(&output, sizeof(expect));
      cudaArray *array;
      cudaMalloc3DArray(&array, &desc, {w, h, d});
      cudaMemcpy3DPeerParms pp = {0};
      pp.srcArray = src;
      pp.srcPos = make_cudaPos(2, 1, 1);
      pp.srcDevice = 0;
      pp.dstArray = array;
      pp.dstPos = make_cudaPos(2, 1, 1);
      pp.dstDevice = 0;
      pp.extent = make_cudaExtent(w - 2, h - 1, d - 1);
      cudaMemcpy3DPeer(&pp);
      auto tex = getTex(array);
      kernel<<<1, 1>>>(output, tex, w, h, d);
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
      for (int i = 0; i < w * h * d; ++i) {
        if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
            output[4 * i + 2] != expect[i].z ||
            output[4 * i + 3] != expect[i].w) {
          pass = false;
          break;
        }
      }
      checkResult("cudaMemcpy3DPeer:a2a:5", pass);
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
      pass = true;
    }
    cudaFreeArray(src);
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
