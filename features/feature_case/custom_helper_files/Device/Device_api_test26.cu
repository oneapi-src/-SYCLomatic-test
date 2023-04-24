// ====------ Device_api_test26.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test26_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test26_out

// CHECK: 13
// TEST_FEATURE: Device_device_info_set_global_mem_size
// TEST_FEATURE: Device_device_info_set_integrated
// TEST_FEATURE: Device_device_info_set_local_mem_size
// TEST_FEATURE: Device_device_info_set_major_version
// TEST_FEATURE: Device_device_info_set_max_clock_frequency
// TEST_FEATURE: Device_device_info_set_max_compute_units
// TEST_FEATURE: Device_device_info_set_max_sub_group_size
// TEST_FEATURE: Device_device_info_set_max_work_group_size
// TEST_FEATURE: Device_device_info_set_max_work_items_per_compute_unit
// TEST_FEATURE: Device_device_info_set_minor_version

// array type is not assignable
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_name
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_max_nd_range_size
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_max_work_item_sizes

int main() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int memoryClockRate = deviceProp.memoryClockRate;
  int memoryBusWidth = deviceProp.memoryBusWidth;
  int minor = deviceProp.minor;
  int multiProcessorCount = deviceProp.multiProcessorCount;
  return 0;
}
