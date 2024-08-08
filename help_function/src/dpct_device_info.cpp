// ====------ dpct_device_info.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <iostream>
#include <dpct/dpct.hpp>

using namespace sycl;

void dev_info_output(const device& dev, std::string selector_name) {
  std::cout << selector_name << ": Selected device: " << dev.get_info<info::device::name>() << "\n";
  std::cout << "            -> Device vendor: " << dev.get_info<info::device::vendor>() << "\n";
}

int main() {
  dpct::get_current_device().queues_wait_and_throw();

  int device_count = dpct::dev_mgr::instance().device_count();
  sycl::queue queue_1(dpct::get_default_context(), dpct::get_current_device());
  if(device_count > 1) {
    dpct::dev_mgr::instance().select_device(1);
    sycl::queue queue_2(dpct::get_default_context(), dpct::get_current_device());
  }

  auto device = dpct::get_default_queue().get_device();

  // To test that all device APIs called in get_device_info() do not break on all
  // devices.
  for (int device_id = 0; device_id < device_count; device_id++) {
    dpct::device_info deviceProp;
    dpct::dev_mgr::instance()
        .get_device(device_id)
        .get_device_info(deviceProp);
  }

  if(device.is_gpu())
      dev_info_output(device, "gpu  device");
  if(device.is_cpu())
      dev_info_output(device, "cpu  device");
  if(device.is_accelerator())
      dev_info_output(device, "accelerator device");

  size_t free_memory, global_memory;
  // Here just to test get_memory_info() to be called, as not all sycl backends
  // can support to query free memory, and return 0 for free memory if the
  // backend does not support it.
  dpct::get_current_device().get_memory_info(free_memory, global_memory);

  printf("Test passed!\n");
  return 0;
}
