// ====------ onedpl_test_group_load.cpp------------ *- C++ -* ----===//

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

template <dpct::group::store_algorithm S>
bool helper_validation_function(const int *ptr, const char *func_name) {
  if constexpr (S == dpct::group::load_algorithm::BLOCK_STORE_DIRECT) {
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
    std::cout << func_name << "_blocked"
              << " pass\n";
  } else {
    for (int i = 0; i < 512; ++i) {
      if (ptr[i] != i) {
        std::cout << func_name << "_striped"
                  << " failed\n";
        std::ostream_iterator<int> Iter(std::cout, ", ");
        std::copy(ptr, ptr + 512, Iter);
        std::cout << std::endl;
        return false;
      }
    }
    std::cout << func_name << "_striped"
              << " pass\n";
  }
  return true;
}

bool subgroup_helper_validation_function(const int *ptr, const uint32_t *sg_sz,
                                         const char *func_name) {
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != i) {
      std::cout << " failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << func_name << " pass\n";
  return true;
}
/*
template < dpct::group::store_algorithm S> bool test_group_store() {
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
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
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
  return helper_validation_function<T>(ptr, "test_group_load_store");
}

/*
bool test_store_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  int num_threads = 128;
  int items_per_thread = 4;
  
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < items_per_thread; ++j) {
      expected[items_per_thread * i + j] =
          (i / sg_sz_val) * sg_sz_val * items_per_thread + sg_sz_val * j +
          i % sg_sz_val;
    }
  }
  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<uint32_t, 1> sg_sz_buf{sycl::range<1>(1)};
  int data_out[512];
  sycl::buffer<int, 1> buffer_out(data_out, 512);

  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc_read(buffer, h, sycl::read_only);
    sycl::accessor dacc_write(buffer_out, h, sycl::write_only);
    sycl::accessor sg_sz_dacc(sg_sz_buf, h, sycl::read_write);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d_r =
              dacc_read.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *d_w =
              dacc_write.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *sg_sz_acc =
              sg_sz_dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          size_t gid = item.get_global_linear_id();
          if (gid == 0) {
            sg_sz_acc[0] = item.get_sub_group().get_local_linear_range();
          }
          dpct::group::store_subgroup_striped<4, int>(item, d_r, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index =
              (item.get_group(2) * item.get_local_range().get(2)) +
              item.get_local_id(2); // Each thread_data has 4 elements
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            dacc_write[global_index * 4 + i] = thread_data[i];
          }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer_out, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  sycl::host_accessor data_accessor_sg(sg_sz_buf, sycl::read_only);
  const uint32_t *ptr_sg =
      data_accessor_sg.get_multi_ptr<sycl::access::decorated::yes>();
  return subgroup_helper_validation_function(
      ptr, ptr_sg, "test_subgroup_striped_standalone");
}
*/
template <dpct::group::load_algorithm T, dpct::group::store_algorithm S> bool test_group_store_standalone() {
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
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
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
  return helper_validation_function<T>(ptr, "test_group_load_store");
}
*/
int main() {

  return !(
      // Calls test_group_load with blocked and striped strategies , should pass
      // both results.
      //test_group_load_store<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>() &&
      //test_group_load_store<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>() &&
      // Calls test_load_subgroup_striped_standalone and should pass
      test_store_subgroup_striped_standalone() );//&
      // Calls test_group_load_standalone with blocked and striped strategies as
      // free functions, should pass both results.
      //test_group_load_store_standalone<
      //    dpct::group::load_algorithm::BLOCK_LOAD_STRIPED, dpct::group::store_algorithm::BLOCK_STORE_STRIPED>() &&
      //test_group_load_store_standalone<
      //    dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, dpct::group::store_algorithm::BLOCK_STORE_DIRECT>());
}
