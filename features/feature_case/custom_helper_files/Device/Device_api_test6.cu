// ====------ Device_api_test6.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test6_out

// CHECK: 30
// TEST_FEATURE: Device_device_ext_get_device_info_return_info
#include<string>
int main() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int id = deviceProp.pciDeviceID;
  std::string name = deviceProp.name;
  return 0;
}

