// ====------ asm_fence.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>

__device__ inline void fence(int *arr, int *lock, bool reset) {
  
  if(threadIdx.x < 10){
    arr[threadIdx.x] = threadIdx.x * 2;
  }
  __syncthreads();
  
  if(threadIdx.x == 0){
  
    if(reset){
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile("fence.acq_rel.gpu;\n");
    
    lock[0] = val;
  }
  
}

__global__ void kernel(int *arr, int *lock, bool reset) {
  fence(arr, lock, reset);
}

int main() {

  int *lock;
  cudaMallocManaged(&lock, sizeof(int));
  lock[0] = 1;
  
  int *arr, *brr;
  cudaMallocManaged(&arr, sizeof(int) * 10);
  cudaMemset(arr, 0, sizeof(int) * 10);
  
  kernel<<<1, 10>>>(arr, lock, false);
  
  cudaDeviceSynchronize();
  
  int res = 0;
  for (int i = 0; i < 10; ++i) {
    if (arr[i] !=  i*2){
        res = 1;
    }
  }
  cudaFree(arr);
  return res;
}
