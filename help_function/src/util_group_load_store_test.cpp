// ====------ util_group_load_store_test.cpp------------ *- C++ -* ----===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>

bool helper_validation_function(const int *ptr, const char *func_name) {
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != i) {
      std::cout << func_name << "_blocked"
                << " failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << func_name << " pass\n";

  return true;
}

template <dpct::group::load_algorithm T, dpct::group::store_algorithm S> bool test_group_load_store(sycl::nd_range<3> &range, char *func_name) {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT 
  // dpct::group::load_algorithm::BLOCK_LOAD_STRIPED in its entirety as API
  // functions
  // Tests dpct::group::store_algorithm::BLOCK_STORE_DIRECT 
  // dpct::group::store_algorithm::BLOCK_STORE_STRIPED in its entirety as API
  // functions
  sycl::queue q(dpct::get_default_queue());
  oneapi::dpl::counting_iterator<int> count_it(0);
  sycl::buffer<int, 1> buffer(count_it, count_it + 512);
  
  q.submit([&](sycl::handler &h) {
    using group_load =
        dpct::group::workgroup_load<4, T, int, const int *, sycl::nd_item<3>>;
    using group_store =
        dpct::group::workgroup_store<4, S, int, int *, sycl::nd_item<3>>;
    size_t temp_storage_size = group_load::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), h);
    sycl::accessor data_accessor_read_write(buffer, h, sycl::read_write);
    h.parallel_for(
        range,
        //sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d_r_w =
              data_accessor_read_write.get_multi_ptr<sycl::access::decorated::yes>()
                  .get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          // Load thread_data of each work item to blocked arrangement
          group_load(tmp).load(item, d_r_w, thread_data);
          // Store thread_data of each work item from blocked arrangement
          group_store(tmp).store(item, d_r_w, thread_data);
          });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_write);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function(ptr, func_name);
}

bool test_load_store_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  for (int i = 0; i < 512; i++)
    data[i] = i;
  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<uint32_t, 1> sg_sz_buf{sycl::range<1>(1)};
  
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc_read_write(buffer, h, sycl::read_write);
    sycl::accessor sg_sz_dacc(sg_sz_buf, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d_r_w =
              dacc_read_write.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *sg_sz_acc =
              sg_sz_dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          size_t gid = item.get_global_linear_id();
          if (gid == 0) {
            sg_sz_acc[0] = item.get_sub_group().get_local_linear_range();
          }
          dpct::group::uninitialized_load_subgroup_striped<4, int>(item, d_r_w,
                                                                   thread_data);
          dpct::group::store_subgroup_striped<4, int>(item, d_r_w, thread_data);
          //call destructor of thread type
          for (size_t i = 0; i < 4; ++i) {
            thread_data[i].~int();  
          }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  sycl::host_accessor data_accessor_sg(sg_sz_buf, sycl::read_only);
  const uint32_t *ptr_sg =
      data_accessor_sg.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function(
      ptr, "test_subgroup_striped_standalone");
}

template <dpct::group::load_algorithm T, dpct::group::store_algorithm S> bool test_group_load_store_standalone(sycl::nd_range<3> & range, char *func_name) {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT &
  // dpct::group::load_algorithm::BLOCK_LOAD_STRIPED as standalone methods
  // Tests dpct::group::store_algorithm::BLOCK_STORE_DIRECT &
  // dpct::group::store_algorithm::BLOCK_STORE_STRIPED as standalone methods
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  for (int i = 0; i < 512; i++)
    data[i] = i;
  sycl::buffer<int, 1> buffer(data, 512);
  
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc_read_write(buffer, h, sycl::read_write);
    h.parallel_for(
        range,
        //sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d_r_w =
              dacc_read_write.get_multi_ptr<sycl::access::decorated::yes>().get();
          // Load thread_data of each work item to blocked arrangement
          if (T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT) {
            dpct::group::load_blocked<4, int>(item, d_r_w, thread_data);
          } else {
            dpct::group::load_striped<4, int>(item, d_r_w, thread_data);
          }
          // Store thread_data of each work item from blocked arrangement
          if (S == dpct::group::store_algorithm::BLOCK_STORE_DIRECT) {
            dpct::group::store_blocked<4, int>(item, d_r_w, thread_data);
          } else {
            dpct::group::store_striped<4, int>(item, d_r_w, thread_data);
          }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function(ptr, func_name);
}

int main() {
  sycl::range<3> global_range{1, 1, 128};
  sycl::range<3> local_range{1, 1, 128};
  sycl::nd_range<3> range{global_range, local_range};
  sycl::range<3> global_range_multi{2, 2, 64};
  sycl::range<3> local_range_multi{1, 1, 64};
  sycl::nd_range<3> range_multi{global_range_multi, local_range_multi};

  return !(
      // Calls test_group_load with blocked and striped strategies , should pass
      // both results.
      test_group_load_store<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>(range, "test_group_load_store") &&
      test_group_load_store<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>(range, "test_group_load_store") &&
      // Calls test_load_subgroup_striped_standalone and should pass
      test_load_store_subgroup_striped_standalone() &&
      // Calls test_group_load_standalone with blocked and striped strategies as
      // free functions, should pass both results.
      test_group_load_store_standalone<
          dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>(range, "test_group_load_store_standalone") &&
      test_group_load_store_standalone<
          dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>(range, "test_group_load_store_standalone") &&
    
    test_group_load_store<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>(range_multi, "test_group_load_store_multiple_wgs") &&
      test_group_load_store_multiple_wgs<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>(range_multi, "test_group_load_store_multiple_wgs") &&
      // Calls test_load_subgroup_striped_standalone and should pass
      test_load_store_subgroup_striped_standalone_multiple_wgs() &&
      // Calls test_group_load_standalone with blocked and striped strategies as
      // free functions, should pass both results.
      test_group_load_store_standalone<
          dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>(range_multi, "test_group_load_store_standalone_multiple_wgs") &&
      test_group_load_store_standalone<
          dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>(range_multi, "test_group_load_store_standalone_multiple_wgs"));
}
