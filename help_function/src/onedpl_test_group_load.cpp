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
bool test_load_blocked_striped() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT & dpct::group::load_algorithm::BLOCK_LOAD_STRIPED 
  // in its entirety as API functions
  sycl::queue q;
  oneapi::dpl::counting_iterator<int> count_it(0);
  sycl::buffer<int, 1> buffer(count_it, count_it + 512);
  
  q.submit([&](sycl::handler &h) {
    sycl::accessor data_accessor(buffer, h, sycl::read_write);
    int thread_data[4];
    using group_load = dpct::group::workgroup_load<4, T, int,  sycl::accessor<int, 3, sycl::access_mode::read_write, sycl::access::target::device>, sycl::nd_item<3>>;
    size_t temp_storage_size = group_load::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), h);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          auto *d = data_accessor.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          group_load(tmp).load(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
            for (int i = 0; i < 4; ++i) {
                data_accessor[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();
  
  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  for (int i = 0; i < 512; ++i) {
    if (ptr[i] != i) {
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

bool test_load_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
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
          dpct::group::uninitialized_load_subgroup_striped<4, int>(item, d, thread_data);
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
          for (int i = 0; i < 4; ++i) {
                dacc[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  int expected[512];
  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i;
    expected[4 * i + 1] = 4 * i + 1;
    expected[4 * i + 2] = 4 * i + 2;
    expected[4 * i + 3] = 4 * i + 3;
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
bool test_load_blocked_striped_standalone() {
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
            {dpct::group::load_blocked<4, T, int>(item, d, thread_data);}
          else
            {dpct::group::load_striped<4, T, int>(item, d, thread_data);}
          // Write thread_data of each work item at index to the global buffer
          int global_index = item.get_global_linear_id() * 4; // Each thread_data has 4 elements
            for (int i = 0; i < 4; ++i) {
                dacc[global_index + i] = thread_data[i];
            }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer, sycl::read_only);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  if(T == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT)
   {
     for (int i = 0; i < 512; ++i) {
      if (ptr[i] != i) {
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
  else{
    int expected[512];
    
    for (int i = 4; i < 128; ++i) {
        expected[4 * i + 0] = 4 * i + (128 * 0);
        expected[4 * i + 1] = 4 * i + 1 + (128 * 1);
        expected[4 * i + 2] = 4 * i + 2 + (128 * 2);
        expected[4 * i + 3] = 4 * i + 3 + (128 * 3);
        
    }

    for(int i=0;i<512;i++){std::cout<<expected[i]<< ", ";}
    std::cout<<"END"<<std::endl;
  
    for(int i=0;i<512;i++){std::cout<<ptr[i]<< ", ";}
    std::cout<<"END"<<std::endl;
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
}


int main() {
  
  return !(//test_load_blocked_striped<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>() && test_load_blocked_striped<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && 
  test_load_subgroup_striped_standalone() && 
  test_load_blocked_striped_standalone<dpct::group::load_algorithm::BLOCK_LOAD_STRIPED>() && test_load_blocked_striped_standalone<dpct::group::load_algorithm::BLOCK_LOAD_DIRECT>());
}
