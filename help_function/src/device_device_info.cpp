// ====------ device_device_info.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {

  auto& Device = dpct::get_current_device();
  dpct::device_info Info;

  // test_feature:get_name()
  Info.get_name();

  // test_feature:get_max_work_item_sizes()
  Info.get_max_work_item_sizes();

  // test_feature:get_host_unified_memory()
  Info.get_host_unified_memory();

  // test_feature:get_major_version()
  Info.get_minor_version();

  // test_feature:get_integrated()
  Info.get_integrated();

  // test_feature:get_max_clock_frequency()
  Info.get_max_clock_frequency();

  // test_feature:get_max_compute_units()
  Info.get_max_compute_units();

  // test_feature:get_max_work_group_size()
  Info.get_max_work_group_size();

  // test_feature:get_max_sub_group_size()
  Info.get_max_sub_group_size();

  // test_feature:get_max_work_items_per_compute_unit()
  Info.get_max_work_items_per_compute_unit();

  // test_feature:get_max_nd_range_size()
  Info.get_max_nd_range_size();

  // test_feature:get_global_mem_size()
  Info.get_global_mem_size();

  // test_feature:get_local_mem_size()
  Info.get_local_mem_size();

  // test_feature:get_memory_clock_rate();
  Info.get_memory_clock_rate();

  // test_feature:get_memory_bus_width();
  Info.get_memory_bus_width();

  const char* Name = "DEVNAME";
  // test_feature:set_name(Name)
  Info.set_name(Name);

  // test_feature:set_host_unified_memory(true)
  Info.set_host_unified_memory(true);

  // test_feature:set_major_version(1)
  Info.set_major_version(1);

  // test_feature:set_minor_version(1)
  Info.set_minor_version(1);

  // test_feature:set_integrated(1)
  Info.set_integrated(1);

  // test_feature:set_max_clock_frequency(1000)
  Info.set_max_clock_frequency(1000);

  // test_feature:set_max_compute_units(32)
  Info.set_max_compute_units(32);

  // test_feature:set_global_mem_size(1000)
  Info.set_global_mem_size(1000);

  // test_feature:set_local_mem_size(1000)
  Info.set_local_mem_size(1000);

  // test_feature:set_max_work_group_size(32)
  Info.set_max_work_group_size(32);

  // test_feature:set_max_sub_group_size(16)
  Info.set_max_sub_group_size(16);

  // test_feature:set_max_work_items_per_compute_unit(16)
  Info.set_max_work_items_per_compute_unit(16);

  int SizeArray[3] = {1,2,3};
  // test_feature:set_max_nd_range_size(SizeArray)
  Info.set_max_nd_range_size(SizeArray);

  return 0;
}
