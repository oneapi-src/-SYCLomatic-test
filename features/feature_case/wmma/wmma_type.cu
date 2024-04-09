// ====---------------- wmma_type.cu--------------- *- CUDA -*------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <mma.h>

__global__ void simple_wmma() {
  nvcuda::wmma::layout_t ly = nvcuda::wmma::mem_row_major;
}

int main() {
  simple_wmma<<<1, 1>>>();
  return 0;
}
