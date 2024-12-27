// ===-------- text_experimental_obj_surf.cu ------- *- CUDA -* ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define PRINT_PASS 1

using namespace std;

int passed = 0;
int failed = 0;
float precision = 0.001;

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

template <typename T>
__global__ void surfDwrite(T *input, cudaSurfaceObject_t surf, int width) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  surf1Dwrite(input[x], surf, x * sizeof(T));
}
template <typename T>
__global__ void surfDwrite(T *input, cudaSurfaceObject_t surf, int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  surf2Dwrite(input[x + y * blockDim.x], surf, x * sizeof(T), y);
}

template <typename T>
__global__ void surfDwrite(T *input, cudaSurfaceObject_t surf, int width, int height, int depth) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    surf3Dwrite(input[x + width *y  + width * height *z], surf, x * sizeof(T), y, z);
}
template <class T, class ArrayType>
cudaSurfaceObject_t createSurface(ArrayType &inputArray,int w, int h) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();

  cudaMallocArray(&inputArray, &desc, w, h);
  cudaResourceDesc surfRes;
  cudaSurfaceObject_t surf;
  memset(&surfRes, 0, sizeof(surfRes));
  surfRes.res.array.array = inputArray;
  surfRes.resType = cudaResourceTypeArray;
  cudaCreateSurfaceObject(&surf, &surfRes);
  return surf;
}

template <class T, class ArrayType>
CUsurfObject createSurfaceDriver(ArrayType &inputArray,int w, int h) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();

  cudaMallocArray(&inputArray, &desc, w, h);
  CUDA_RESOURCE_DESC  surfRes;
  CUsurfObject surf;
  memset(&surfRes, 0, sizeof(surfRes));
  surfRes.res.array.hArray = inputArray;
  surfRes.resType = CU_RESOURCE_TYPE_ARRAY;
  cuSurfObjectCreate(&surf, &surfRes);
  return surf;
}

template <class T, class ArrayType>
cudaSurfaceObject_t createSurface(ArrayType &inputArray, cudaExtent & extent) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();

  cudaMalloc3DArray(&inputArray, &desc, extent);
  cudaResourceDesc surfRes;
  cudaSurfaceObject_t surf;
  memset(&surfRes, 0, sizeof(surfRes));
  surfRes.res.array.array = inputArray;
  surfRes.resType = cudaResourceTypeArray;
  cudaCreateSurfaceObject(&surf, &surfRes);
  return surf;
}

// Compare the uint1 type with 1 Dinmesion image.
int test_surface_write_uint1() {
  bool pass = true;

  const int width = 8;
  cudaArray_t inputArray;
  cudaSurfaceObject_t surf = createSurface<uint1>(inputArray, width, 0);
  uint1 expect[width] = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},
  };
  uint1 *inputData;
  cudaMalloc(&inputData, width * sizeof(uint1));
  cudaMemcpy(inputData, expect, width * sizeof(uint1), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, 1, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width);
  cudaDeviceSynchronize();
  unsigned int *output;
  cudaMallocManaged(&output, sizeof(expect));
  cudaMemcpyFromArray(output, inputArray, 0, 0, width * sizeof(uint1), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < width; ++i) {
    if (output[i] != expect[i].x) {
      pass = false;
      break;
    }
  }
  checkResult("surface_write_uint1", pass);
  if (!pass) {
    for (int i = 0; i < width; ++i)
      cout << "{" << output[i] << "}, ";
    cout << endl;
  }
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  cudaFree(output);
  pass = true;
  return 0;
  
}
// Compare the uint1 type with 1 Dinmesion image through cuda driver API.
int test_surface_driver() {
  const int width = 8;
  CUarray inputArray;
  CUDA_RESOURCE_DESC  surfRes;
  CUsurfObject surf;
  memset(&surfRes, 0, sizeof(surfRes));
  surfRes.res.array.hArray = inputArray;
  surfRes.resType = CU_RESOURCE_TYPE_ARRAY;
  cuSurfObjectCreate(&surf, &surfRes);
  
  cudaDeviceSynchronize();
  cuSurfObjectDestroy(surf);
  CUDA_RESOURCE_DESC pResDesc;
  cuSurfObjectGetResourceDesc(&pResDesc, surf);
  return 0;
}
// 2 element char 1D image.
int test_surface_write_char2() {
  bool pass = true;

  const unsigned int width = 4;

  cudaArray_t inputArray;
  cudaSurfaceObject_t surf = createSurface<char2>(inputArray, width, 0);
  char2 expect[width] = {
        {1, 2}, 
        {3, 4}, 
        {5, 6}, 
        {7, 8},
  };

  char2 *inputData;
  cudaMalloc(&inputData, width * sizeof(char2));
  cudaMemcpy(inputData, expect, width * sizeof(char2), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, 1, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width);
  cudaDeviceSynchronize();
  char *output;
  cudaMallocManaged(&output, sizeof(expect));
  cudaMemcpyFromArray(output, inputArray, 0, 0, width * sizeof(char2), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < width; i++) {
    if (output[2*i] != expect[i].x || output[2*i+1] != expect[i].y) {
      pass = false;
      break;
    }
  }
  checkResult("test_surface_write_char2", pass);
  if (!pass) {
    for (int i = 0; i < width; ++i)
      cout << "{" << (int)output[2*i] << ", " << (int)output[2*i+1] << "}, ";
    cout << endl;
  }
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  cudaFree(output);
  pass = true;
  return 0;
}

// 1 element int1 2D image.
int test_surface_write_int1() {
  bool pass = true;

  const int h = 3;
  const int w = 2;  

  cudaArray_t inputArray;
  cudaSurfaceObject_t surf = createSurface<int1>(inputArray, w, h);
  int1 expect[h * w] = {
        {1}, {2}, 
        {3}, {4},
        {5}, {6},
  };
  int1 *inputData;
  cudaMalloc(&inputData, h * w * sizeof(int1));
  cudaMemcpy(inputData, expect, h * w * sizeof(int1), cudaMemcpyHostToDevice);
  dim3 dimBlock(w, h, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, w, h);
  cudaDeviceSynchronize();
  unsigned int *output;
  cudaMallocManaged(&output, sizeof(expect));
  // cudaMemcpyFromArray(output, inputArray, 0, 0, h * w * sizeof(int1), cudaMemcpyDeviceToHost);
  cudaMemcpy2DFromArray(output, sizeof(int1)*w, inputArray, 0,0, w * sizeof(int1), h,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < h * w; ++i) {
    if (output[i] != expect[i].x) {
      pass =false;
      break;
    }
  }

  checkResult("test_surface_write_int1", pass);
  if (!pass) {
    for (int i = 0; i < h * w; ++i)
      cout << "{" << output[i] << "}, ";
    cout << endl;
  }
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  cudaFree(output);
  pass = true;
  return 0;
}

// 1 element short1 3D image.
int test_surface_write_short1() {
  bool pass = true;

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<short1>();
  const unsigned int width = 2;
  const unsigned int height = 3;
  const unsigned int depth = 3;
  cudaArray_t inputArray;
  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaSurfaceObject_t surf = createSurface<short1>(inputArray, extent);
  short1 expect[width * height * depth]= {
    {1}, {2},
    {3}, {4},
    {5}, {6},

    {7}, {8},
    {9}, {10},
    {11}, {12},

    {13}, {14},
    {15}, {16},
    {17}, {18},
  };
  short1 *inputData;
  cudaMalloc(&inputData, width * height * depth * sizeof(short1));
  cudaMemcpy(inputData, expect, width * height * depth * sizeof(short1), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, height, depth);
  dim3 dimGrid(1, 1, 1);
  #if 1
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width, height, depth); 
  #else
  cudaMemcpyToArray(inputArray, 0, 0, expect, sizeof(short1) * width * height * depth, cudaMemcpyHostToDevice);
  #endif
  cudaDeviceSynchronize();
  short1 *hostData = new short1[width * height * depth];
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = inputArray;
  copyParams.dstPtr = make_cudaPitchedPtr(hostData, width * sizeof(short1), width, height);
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyDeviceToHost;


  cudaMemcpy3D(&copyParams);

  cudaDeviceSynchronize();
  for (int i = 0; i < width * height * depth; ++i) {
    if (hostData[i].x != expect[i].x) {
      pass = false; 
      break;
    }
  }
  checkResult("test_surface_write_short1", pass);
  if (!pass) {
    for (int i = 0; i < width * height * depth; ++i)
      cout << "{" << hostData[i].x << "}, ";
    cout << endl;
  }
  delete[] hostData;
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  cudaFree(inputData);
  pass = true;
  return 0;
}
// 2 element char 2D image.
int test_surface_write_uchar2() {
  bool pass = true;
  const unsigned int width = 2;
  const unsigned int height = 3;

  cudaArray_t inputArray;
  cudaSurfaceObject_t surf = createSurface<uchar2>(inputArray, width, height);

  uchar2 expect[width * height] = {
        {1, 2}, {3, 4}, 
        {5, 6}, {7, 8},
        {9, 10}, {11, 12},
  };

  uchar2 *inputData;
  cudaMalloc(&inputData, width * height * sizeof(uchar2));
  cudaMemcpy(inputData, expect, width * height * sizeof(uchar2), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, height, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width, height);
  cudaDeviceSynchronize();
  char *output;
  cudaMallocManaged(&output, sizeof(expect));
  // cudaMemcpyFromArray(output, inputArray, 0, 0, width * height * sizeof(char2), cudaMemcpyDeviceToHost);
    cudaMemcpy2DFromArray(output, sizeof(char2)*width, inputArray, 0, 0, width * sizeof(char2), height,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < width * height; i++) {
    if (output[2*i] != expect[i].x || output[2*i+1] != expect[i].y) {
      pass = false;
      break;
    }
  }
  checkResult("test_surface_write_uchar2", pass);
  if (!pass) {
    for (int i = 0; i < width * height; ++i)
      cout << "{" << (int)output[2*i] << ", " << (int)output[2*i+1] << "}, ";
    cout << endl;
  }
  cudaFree(output);
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  pass = true;
  return 0;
}



// 2 element ushort2 3D image.
int test_surface_write_ushort2() {
  bool pass = true;

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort2>();
  const unsigned int width = 2;
  const unsigned int height = 3;
  const unsigned int depth = 3;
  cudaArray_t inputArray;
  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaSurfaceObject_t surf = createSurface<ushort2>(inputArray, extent);

  ushort2 expect[width * height * depth]= {
    {1, 2}, {3, 4},
    {5, 6}, {7, 8},
    {9, 10}, {11, 12}, 
    
    {13, 14}, {15, 16},
    {17, 18}, {19, 20},
    {21, 22}, {23, 24},
    
    {25, 26}, {27, 28},
    {29, 30}, {31, 32},
    {33, 34}, {35, 36},
  };
  ushort2 *inputData;
  cudaMalloc(&inputData, width * height * depth * sizeof(ushort2));
  cudaMemcpy(inputData, expect, width * height * depth * sizeof(ushort2), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, height, depth);
  dim3 dimGrid(1, 1, 1);
  #if 1
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width, height, depth); 
  #else
  cudaMemcpyToArray(inputArray, 0, 0, expect, sizeof(ushort2) * width * height * depth, cudaMemcpyHostToDevice);
  #endif
  cudaDeviceSynchronize();
  unsigned short *hostData = new unsigned short[sizeof(expect)];
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = inputArray;
  copyParams.dstPtr = make_cudaPitchedPtr(hostData, width * sizeof(ushort2), width, height);
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&copyParams);

  cudaDeviceSynchronize();
  for (int i = 0; i < width * height * depth; ++i) {
    if (hostData[2 * i] != expect[i].x || hostData[2 * i + 1] != expect[i].y) {
      pass = false;
      break;
    }
  }
  checkResult("test_surface_write_ushort2", pass);
  if (!pass) {
    for (int i = 0; i < width * height * depth; ++i)
      cout << "{" << hostData[2 * i] << ", " << hostData[2 * i + 1] << "}, ";
    cout << endl;
  }
  delete []hostData;
  cudaFree(inputData);
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  pass = true;
  return 0;
}

// 4 element char 1D image.
int test_surface_write_float4() {
  bool pass = true;

  const unsigned int width = 4;

  cudaArray_t inputArray;
  cudaSurfaceObject_t surf = createSurface<float4>(inputArray, width, 0);

  float4 expect[width] = {
    {1, 2, 3, 4}, 
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16}
  };

  float4 *inputData;
  cudaMalloc(&inputData, width * sizeof(float4));
  cudaMemcpy(inputData, expect, width * sizeof(float4), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, 1, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width);
  cudaDeviceSynchronize();
  float *output;

  cudaMallocManaged(&output, sizeof(expect));
  cudaMemcpyFromArray(output, inputArray, 0, 0, sizeof(expect), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < width; i++) {
    if (fabs(output[4*i] + output[4*i + 1] + output[4 *i + 2] + output[4*i +3] - expect[i].x - expect[i].y - expect[i].z - expect[i].w) > precision) {
      pass =false;
      break;
    }
  }

  checkResult("test_surface_write_float4", pass);
  if (!pass) {
      for (int i = 0; i < width; i++) {
      cout << "{" << output[4 * i] << ", " << output[4 * i + 1] <<", " << output[4 * i + 2] <<", " << output[4 * i + 3] << "}, ";
      cout << endl;
      }
  }
  cudaFree(output);
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  pass = true;
  return 0;
}
// 4 element char 2D image.
int test_surface_write_int4() {
  bool pass = true;

  const unsigned int width = 2;
  const unsigned int height = 3;

  cudaArray_t inputArray;

  cudaSurfaceObject_t surf =createSurface<int4>(inputArray, width, height);

  int4 expect[width * height] = {
    {1, 2, 3, 4}, {5, 6, 7, 8},
    {9, 10, 11, 12}, {13, 14, 15, 16},
    {17, 18, 19, 20}, {21, 22, 23, 24}};

  int4 *inputData;
  cudaMalloc(&inputData, width * height * sizeof(int4));
  cudaMemcpy(inputData, expect, width * height * sizeof(int4), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, height, 1);
  dim3 dimGrid(1, 1, 1);
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width, height);
  cudaDeviceSynchronize();
  int *output;
  cudaMallocManaged(&output, sizeof(expect));
  // cudaMemcpyFromArray(output, inputArray, 0, 0, width * height * sizeof(int4), cudaMemcpyDeviceToHost);
    cudaMemcpy2DFromArray(output, sizeof(int4)*width, inputArray, 0, 0, width * sizeof(int4), height,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < width * height; i++) {

    if (output[4*i] != expect[i].x || output[4*i+1] != expect[i].y || output[4*i + 2] != expect[i].z || output[4*i + 3] != expect[i].w) {
      pass = false;
      break;
    }
  }

  checkResult("test_surface_write_int4", pass);
  if (!pass) {
      for (int i = 0; i < width * height; i++) {
      cout << "{ " << output[4*i] << " ," << output[4*i+1] << " , " << output[4*i+2] << " , " << output[4*i+3] << "} " << endl;
      }
  }
  cudaFree(output);
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  pass = true;
  return 0;
}



// 4 element uint4 3D image.
int test_surface_write_uint4() {
  bool pass = true;

  const unsigned int width = 2;
  const unsigned int height = 3;
  const unsigned int depth = 3;
  cudaArray_t inputArray;
  cudaExtent extent = make_cudaExtent(width, height, depth);

  cudaSurfaceObject_t surf = createSurface<uint4>(inputArray, extent);
  uint4 expect[width * height * depth]= {
        {1, 2, 3, 4},     {5, 6, 7, 8},     // 1.1
        {9, 10, 11, 12},  {13, 14, 15, 16}, // 1.2
        {17, 18, 19, 20}, {21, 22, 23, 24}, // 1.3

        {25, 26, 27, 28}, {29, 30, 31, 32}, // 2.1
        {33, 34, 35, 36}, {37, 38, 39, 40}, // 2.2
        {41, 42, 43, 44}, {45, 46, 47, 48}, // 2.3

        {49, 50, 51, 52}, {53, 54, 55, 56}, // 2.4
        {57, 58, 59, 60}, {61, 62, 63, 64}, // 2.5
        {65, 66, 67, 68}, {69, 70, 71, 72}, // 2.6
  };
  uint4 *inputData;
  cudaMalloc(&inputData, width * height * depth * sizeof(uint4));
  cudaMemcpy(inputData, expect, width * height * depth * sizeof(uint4), cudaMemcpyHostToDevice);
  dim3 dimBlock(width, height, depth);
  dim3 dimGrid(1, 1, 1);
  #if 1
  surfDwrite<<<dimGrid, dimBlock>>>(inputData, surf, width, height, depth); 
  #else
  cudaMemcpyToArray(inputArray, 0, 0, expect, sizeof(uint4) * width * height * depth, cudaMemcpyHostToDevice);
  #endif
  cudaDeviceSynchronize();
  unsigned int *hostData = new unsigned int[width * height * depth * 4];
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = inputArray;
  copyParams.dstPtr = make_cudaPitchedPtr(hostData, width * sizeof(uint4), width, height);
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&copyParams);

  cudaDeviceSynchronize();
  for (int i = 0; i < width * height * depth; ++i) {
     
    if (hostData[4 * i] != expect[i].x || hostData[4 * i + 1] != expect[i].y ||
        hostData[4 * i + 2] != expect[i].z || hostData[4 * i + 3] != expect[i].w) {
      pass = false;
      break;
    }
  }
  checkResult("test_surface_write_uint4", pass);
  if (!pass) {
      for (int i = 0; i < width * height; i++) {
      cout << "{ " << hostData[4*i] << " ," << hostData[4*i+1] << " , " << hostData[4*i+2] << " , " << hostData[4*i+3] << "} " << endl;
      }
  }
  // cudaFree(output);
  delete []hostData;
  cudaFree(inputData);
  pass = true;
  cudaDestroyTextureObject(surf);
  cudaFreeArray(inputArray);
  return 0;
}

int main() {
  bool pass = true;
  {// Test the surface write function with different data type.
  test_surface_write_uint1();
  // test_surface_driver();
  test_surface_write_char2();
  test_surface_write_int1();
  test_surface_write_short1();
  test_surface_write_uchar2();
  test_surface_write_ushort2();
  test_surface_write_float4();
  test_surface_write_int4();
  test_surface_write_uint4();
  }
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
