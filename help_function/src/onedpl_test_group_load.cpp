// ====------ onedpl_test_group_load.cpp------------ *- C++ -* ----===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>
#include <oneapi/dpl/iterator>


template<dpct::group::load_algorithm T>
bool helper_validation_function(const int* ptr, const char * func_name){
  if constexpr ( T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
  {
     for (int i = 0; i < 512; ++i) {
      if (ptr[i] != i) {
        std::cout << func_name << "_blocked" <<" failed\n";
        std::ostream_iterator<int> Iter(std::cout, ", ");
        std::copy(ptr, ptr + 512, Iter);
        std::cout << std::endl;
        return false;
        }
     }
     std::cout << func_name << "_blocked" <<" pass\n";
    
  }
    
  else{
    int expected[512];
    int num_threads = 128;
    int items_per_thread = 4;
    for (int i = 0;i < num_threads; ++i){
        for(int j=0;j < items_per_thread; ++j){
          expected[i * items_per_thread +j] = j * num_threads +i;  
        }
      }
    for (int i = 0; i < 512; ++i) {
        if (ptr[i] != expected[i]) {
          std::cout << func_name << "_striped" <<" failed\n";
          std::ostream_iterator<int> Iter(std::cout, ", ");
          std::copy(ptr, ptr + 512, Iter);
          std::cout << std::endl;
          return false;
        }
      }
  
    std::cout << func_name << "_striped" <<" pass\n";
   
  }
  return true;
}

bool subgroup_helper_validation_function(const int* ptr,const int &sg_sz, const char* func_name){
  int expected[512];
  int num_threads = 128;
  int items_per_thread = 4;
  for (int i = 0; i < num_threads; ++i) {
      for (int j = 0; j < items_per_thread; ++j) {
          expected[items_per_thread * i + j] = (i / sg_sz) * sg_sz * items_per_thread + sg_sz * j + i % sg_sz;
      }
   }
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != expected[i]) {
      std::cout <<" failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(ptr, ptr + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout <<" pass\n";
  return true;
}

template<dpct::group::load_algorithm T>
bool test_group_load() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT & dpct::group::load_algorithm::BLOCK_LOAD_STRIPED 
  // in its entirety as API functions
  sycl::queue q;
  oneapi::dpl::counting_iterator<int> count_it(0);
  sycl::buffer<int, 1> buffer(count_it, count_it + 512);
  
  q.submit([&](sycl::handler &h) {
    using group_load = dpct::group::workgroup_load<4, T, int,  int *, sycl::nd_item<3>>;
    size_t temp_storage_size = group_load::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), h);
    sycl::accessor data_accessor(buffer, h, sycl::read_write);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = data_accessor.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          group_load(tmp).load(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_group(2)*item.get_local_range().get(2) + item.get_local_id(2); // Each thread_data has 4 elements
          #pragma unroll
            for (int i = 0; i < 4; ++i) {
                data_accessor[global_index * 4 + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();
  
  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function<T>(ptr, "test_group_load");
}

bool test_load_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 512; i++) data[i] = i;

  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<uint32_t> sg_sz_buf(1);
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto sg_sz_acc = sg_sz_buf.get_access<sycl::access::mode::read_write>.(h);
          size_t gid = item.get_global_linear_id();
          if (gid == 0) {
                sg_sz_acc[0] = item.get_sub_group().get_local_linear_range();
          }
          dpct::group::uninitialized_load_subgroup_striped<4, int>(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_group(2)*item.get_local_range().get(2) + item.get_local_id(2); // Each thread_data has 4 elements
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
                dacc[global_index * 4 + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  auto sg_sz = sg_sz_acc.get_host_access()[0];
  return subgroup_helper_validation_function(ptr, sg_sz, "test_subgroup_striped_standalone");
}

template<dpct::group::load_algorithm T>
bool test_group_load_standalone() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT & dpct::group::load_algorithm::BLOCK_LOAD_STRIPED 
  // as standalone methods
  sycl::queue q;
  int data[512];
  for (int i = 0; i < 512; i++) data[i] = i;

  sycl::buffer<int, 1> buffer(data, 512);
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          auto *d = dacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          if( T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
            {dpct::group::load_blocked<4, int>(item, d, thread_data);}
          else
            {dpct::group::load_striped<4, int>(item, d, thread_data);}
          // Write thread_data of each work item at index to the global buffer
          int global_index = int global_index = item.get_group(2)*item.get_local_range().get(2) + item.get_local_id(2); // Each thread_data has 4 elements
          #pragma unroll
            for (int i = 0; i < 4; ++i) {
                dacc[global_index * 4 + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function<T>(ptr, "test_group_load");
}


int main() {
  
  return !(test_group_load<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>() && test_group_load<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && test_load_subgroup_striped_standalone() && 
  test_group_load_standalone<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && test_group_load_standalone<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>());
}