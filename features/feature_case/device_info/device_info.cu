// ====------ device_info.cu---------- *- CUDA -* -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------===//

#include <iostream>
#include <stdio.h>

void test0() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  printf("total_mem : [%lu]\n", total_mem);
  printf("free_mem  : [%lu]\n", free_mem);
}

void test1() {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);

  const int id = properties.pciDeviceID;
  const cudaUUID_t uuid = properties.uuid;
  auto maxTexture1D = properties.maxTexture1D;
  auto maxTexture2D = properties.maxTexture2D;
  auto maxTexture3D = properties.maxTexture3D;
  auto regsPerBlock = properties.regsPerBlock;

  std::cout << "Device ID: " << id << std::endl;
  std::cout << "Device UUID: ";
  for (int i = 0; i < 16; i++) {
    std::cout << std::hex
              << static_cast<int>(static_cast<unsigned char>(uuid.bytes[i]))
              << " ";
  }
  std::cout << std::endl;
  std::cout << "Device maxTexture1D: " << maxTexture1D << std::endl;
  std::cout << "Device maxTexture2D: " << maxTexture2D[0] << " "
            << properties.maxTexture2D[1] << std::endl;
  std::cout << "Device maxTexture3D: " << maxTexture3D[0] << " "
            << properties.maxTexture3D[1] << " " << maxTexture3D[2]
            << std::endl;
  std::cout << "Device regsPerBlock: " << regsPerBlock << std::endl;
}

int main() {
  test0();
  test1();
}
