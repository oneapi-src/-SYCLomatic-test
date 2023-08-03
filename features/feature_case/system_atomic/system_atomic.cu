// ====------------------------ system_atomic.cu----------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <stdio.h>
typedef unsigned int dataType;
#define ArraySize 11
#define numThreads 256
#define numBlocks 64

const dataType Limit = 139;

__global__ void atomic_test_kernel(dataType *ddata) {

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // add test
  atomicAdd_system(&ddata[0], 1);
  // sub test
  atomicSub_system(&ddata[1], 1);
  // exchange test
  atomicExch_system(&ddata[2], tid);
  // max test
  atomicMax_system(&ddata[3], tid);
  // min test
  atomicMin(&ddata[4], tid);
  // CAS test
  atomicCAS_system(&ddata[5], tid - 1, tid);
  // And TEst
  atomicAnd_system(&ddata[6], tid + 7);
  // Or test
  if (tid < 32)
    atomicOr_system(&ddata[7], 1 << tid);
  // Xor test
  atomicXor_system(&ddata[8], tid);
  // Inc test
  atomicInc_system(&ddata[9], 0);
  // Dec test
  atomicDec_system(&ddata[10], Limit);
}

int main(int argc, char **argv) {

  int err = 0;

  dataType Hdata[ArraySize];
  dataType Hdata2[ArraySize];

  printf("atomic test \n");

  Hdata[0] = 0;                      // add
  Hdata[1] = numThreads * numBlocks; // sub
  Hdata[2] = 0;                      // exchange
  Hdata[3] = 0;                      // max
  Hdata[4] = 0;                      // min
  Hdata[5] = 0;                      // CAS
  Hdata[6] = 0xff;                   // And
  Hdata[7] = 0;                      // or
  Hdata[8] = 0xff;                   // xor
  Hdata[9] = 10;                     // inc
  Hdata[10] = 0;                     // dec

  // allocate device memory for result
  dataType *Ddata;
  cudaMalloc((void **)&Ddata, ArraySize * sizeof(dataType));
  cudaMemcpy(Ddata, Hdata, ArraySize * sizeof(dataType),
             cudaMemcpyHostToDevice);
  atomic_test_kernel<<<numBlocks, numThreads>>>(Ddata);
  cudaMemcpy(Hdata2, Ddata, ArraySize * sizeof(dataType),
             cudaMemcpyDeviceToHost);

  // check add
  if (Hdata2[0] != (numThreads * numBlocks)) {
    err = -1;
    printf("atomicAdd test failed\n");
  }
  // check sub
  if (Hdata2[1] != 0) {
    err = -1;
    printf("atomicSub test failed\n");
  }
  // check exchange
  if (!(Hdata2[2] >= 0 && Hdata2[2] < (numThreads * numBlocks))) {
    err = -1;
    printf("atomicExch test failed: %d,%d\n", Hdata2[2],
           numThreads * numBlocks);
  }
  // check max
  if (Hdata2[3] != (numThreads * numBlocks - 1)) {
    err = -1;
    printf("atomicMax test failed, %d %d\n", Hdata2[3],
           numThreads * numBlocks - 1);
  }
  // check min
  if (Hdata2[4] != 0) {
    err = -1;
    printf("atomicMin test failed, %d \n", Hdata2[4]);
  }
  // check CAS
  if (!(Hdata2[5] > 0 && Hdata2[5] <= (numThreads * numBlocks - 1))) {
    err = -1;
    printf("atomicCAS test failed,%d\n", Hdata2[5]);
  }
  // check And
  if (Hdata2[6] != 0) {
    err = -1;
    printf("atomicAnd test failed, %d\n", Hdata2[6]);
  }
  // check or
  if (Hdata2[7] != -1) {
    err = -1;
    printf("atomicOr test failed, %d\n", Hdata2[7]);
  }
  // check Xor Hdata2[8]
  if (Hdata2[8] != 255) {
    err = -1;
    printf("atomicXor test failed , %d \n", Hdata2[8]);
  }
  // check Inc Hdata2[9]
  if (Hdata2[9] != 0) {
    err = -1;
    printf("atomicInc test failed , %d \n", Hdata2[9]);
  }

  dataType val = 0;
  for (int i = 0; i < 256 * 64; ++i) {
    val = ((val == 0) || (val > Limit)) ? Limit : val - 1;
  }
  // check dec Hdata2[10]
  if (Hdata2[10] != val) {
    err = -1;
    printf("atomicDec test failed , %d \n", Hdata2[10]);
  }

  cudaFree(Ddata);
  printf("atomic test completed, returned %s\n", err == 0 ? "OK" : "ERROR");
  return err;
}
