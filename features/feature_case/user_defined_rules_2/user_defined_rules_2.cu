// ===------ user_defined_rules_2.cu ---------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

__global__ void foo1_kernel() {}
void foo1() {
  foo1_kernel<<<1, 1>>>();
}

__global__ void foo2_kernel(double *d) {}

void foo2() {
  double *d;
  cudaMalloc(&d, sizeof(double));
  foo2_kernel<<<1, 1>>>(d);
  cudaFree(d);
}

int main(){
  foo1();
  foo2();
  cudaDeviceSynchronize();
  return 0;
}
