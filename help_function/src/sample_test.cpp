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
#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif


namespace dpct{
namespace group{

enum load_algorithm {

  BLOCK_LOAD_DIRECT,
  BLOCK_LOAD_STRIPED,
  // To-do: BLOCK_LOAD_WARP_TRANSPOSE

};

// loads a linear segment of workgroup items into a blocked arrangement.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename InputIteratorT, typename Item>
 __dpct_inline__ void load_blocked(const Item &item, InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  size_t linear_tid = item.get_local_linear_id();
  int ltid = int(linear_tid);
  uint32_t workgroup_offset = linear_tid * ITEMS_PER_WORK_ITEM;
  //static const CONSTANT char FMT[] = "n: %u\n";
  //sycl::ext::oneapi::experimental::printf(FMT,ltid);
#pragma unroll
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    items[idx] = block_itr[workgroup_offset + idx];
  }
}

// loads a linear segment of workgroup items into a striped arrangement.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename InputIteratorT, typename Item>
 __dpct_inline__ void load_striped(const Item &item, InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  size_t linear_tid = item.get_local_linear_id();
  size_t group_work_items = item.get_local_range().size();
  //static const CONSTANT char FMT[] = "n: %u\n";
  //sycl::ext::oneapi::experimental::printf(FMT,linear_tid);
  //sycl::ext::oneapi::experimental::printf("y: %u\n",group_work_items);
  //sycl::ext::oneapi::experimental::printf("items_per_wi: %u\n",ITEMS_PER_WORK_ITEM);
#pragma unroll
  
  
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    items[idx] = block_itr[linear_tid + (idx * group_work_items)];
  }
}

// loads a linear segment of workgroup items into a subgroup striped
// arrangement. Created as free function until exchange mechanism is
// implemented.
// To-do: inline this function with BLOCK_LOAD_WARP_TRANSPOSE mechanism
template <size_t ITEMS_PER_WORK_ITEM, typename InputT, typename InputIteratorT,
          typename Item>
__dpct_inline__ void
uninitialized_load_subgroup_striped(const Item &item, InputIteratorT block_itr,
                                    InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  // This implementation uses unintialized memory for loading linear segments
  // into warp striped arrangement.
  uint32_t subgroup_offset = item.get_sub_group().get_local_linear_id();
  uint32_t subgroup_size = item.get_sub_group().get_local_linear_range();
  uint32_t subgroup_idx = item.get_sub_group().get_group_linear_id();
  uint32_t initial_offset =
      (subgroup_idx * ITEMS_PER_WORK_ITEM * subgroup_size) + subgroup_offset;
#pragma unroll
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    new (&items[idx]) InputT(block_itr[initial_offset + (idx * subgroup_size)]);
  }
}

template <size_t ITEMS_PER_WORK_ITEM, load_algorithm ALGORITHM, typename InputT,
          typename InputIteratorT, typename Item>
class workgroup_load {
public:
  static size_t get_local_memory_size(size_t group_work_items) { return 0; }
  workgroup_load(uint8_t *local_memory) : _local_memory(local_memory) {}

  __dpct_inline__ void load(const Item &item, InputIteratorT block_itr,
                            InputT (&items)[ITEMS_PER_WORK_ITEM]) {

    if constexpr (ALGORITHM == dpct::group::load_algorithm::BLOCK_LOAD_DIRECT) {
      //sycl::ext::oneapi::experimental::printf(" in direct ");
      load_blocked<ITEMS_PER_WORK_ITEM, InputT>(item, block_itr, items);
    } if constexpr (ALGORITHM == BLOCK_LOAD_STRIPED) {
      //sycl::ext::oneapi::experimental::printf(" in striped ");
      load_striped<ITEMS_PER_WORK_ITEM, InputT>(item, block_itr, items);
    }
  }

private:
  uint8_t *_local_memory;
};



enum store_algorithm {

  BLOCK_STORE_DIRECT,
  BLOCK_STORE_STRIPED,
  // To-do: BLOCK_STORE_WARP_TRANSPOSE
  // To-do: BLOCK_STORE_VECTORIZE

};

/// Stores a blocked arrangement of work items linear segment of items.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename OutputIteratorT, typename Item>
__dpct_inline__ void store_blocked(const Item &item, OutputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range storage across
  // workgroup items To-do: Decide whether range storage is required for group
  // storage
  size_t linear_tid = item.get_local_linear_id();
  OutputIteratorT workitem_itr = block_itr + (linear_tid * ITEMS_PER_WORK_ITEM);
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[idx] = items[idx];
  }
}

/// Stores a striped arrangement of work items linear segment of items.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename OutputIteratorT, typename Item>
__dpct_inline__ void store_striped(const Item &item, OutputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range storage across
  // workgroup items To-do: Decide whether range storage is required for group
  // storage
  size_t linear_tid = item.get_local_linear_id();
  OutputIteratorT workitem_itr = block_itr + linear_tid; 
  size_t GROUP_WORK_ITEMS = item.get_global_range().size();
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[(idx * GROUP_WORK_ITEMS)] = items[idx];
  }
}

/// Stores a warp-striped arrangement of work items linear segment of items.
// Created as free function until exchange mechanism is
// implemented.
// To-do: inline this function with BLOCK_STORE_WARP_TRANSPOSE mechanism
template <size_t ITEMS_PER_WORK_ITEM, typename InputT, typename OutputIteratorT,
          typename Item>
__dpct_inline__ void
store_subgroup_striped(const Item &item, OutputIteratorT block_itr,
                                    InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  // This implementation uses unintialized memory for loading linear segments
  // into warp striped arrangement.
  uint32_t subgroup_offset = item.get_sub_group().get_local_linear_id();
  uint32_t subgroup_size = item.get_sub_group().get_local_linear_range();
  uint32_t subgroup_idx = item.get_sub_group().get_group_linear_id();
  uint32_t initial_offset =
      (subgroup_idx * ITEMS_PER_WORK_ITEM * subgroup_size) + subgroup_offset;
  OutputIteratorT workitem_itr = block_itr + initial_offset;
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[(idx * subgroup_size)] = items[idx];
  }
}

// template parameters :
// ITEMS_PER_WORK_ITEM: size_t variable controlling the number of items per
// thread/work_item
// ALGORITHM: store_algorithm variable controlling the type of store operation.
// InputT: type for input sequence.
// OutputIteratorT:  output iterator type
// Item : typename parameter resembling sycl::nd_item<3> .
template <size_t ITEMS_PER_WORK_ITEM, store_algorithm ALGORITHM, typename InputT,
          typename OutputIteratorT, typename Item>
class workgroup_store {
public:
  static size_t get_local_memory_size(size_t group_work_items) { return 0; }
  workgroup_store(uint8_t *local_memory) : _local_memory(local_memory) {}
  
  __dpct_inline__ void store(const Item &item, OutputIteratorT block_itr,
                            InputT (&items)[ITEMS_PER_WORK_ITEM]) {

    if constexpr (ALGORITHM == BLOCK_STORE_DIRECT) {
      store_blocked<ITEMS_PER_WORK_ITEM>(item, block_itr, (&items)[ITEMS_PER_WORK_ITEM]);
    } else if constexpr (ALGORITHM == BLOCK_STORE_STRIPED) {
      store_striped<ITEMS_PER_WORK_ITEM>(item, block_itr, (&items)[ITEMS_PER_WORK_ITEM]);
    }
  }
  
private:
  uint8_t *_local_memory;

};

}
}

template <dpct::group::store_algorithm S>
bool helper_validation_function(const int *ptr, const char *func_name) {
  if constexpr (S == dpct::group::store_algorithm::BLOCK_STORE_DIRECT) {
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
    int expected[512];
    int num_threads = 128;
    int items_per_thread = 4;
    for (int i = 0; i < num_threads; ++i) {
      for (int j = 0; j < items_per_thread; ++j) {
        expected[i * items_per_thread + j] = j * num_threads + i;
      }
    }
    for (int i = 0; i < 512; ++i) {
      if (ptr[i] != expected[i]) {
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
  
  int expected[512];
  int num_threads = 128;
  int items_per_thread = 4;
  uint32_t sg_sz_val = *sg_sz;
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < items_per_thread; ++j) {
      expected[items_per_thread * i + j] =
          (i / sg_sz_val) * sg_sz_val * items_per_thread + sg_sz_val * j +
          i % sg_sz_val;
    }
  }
  
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

template < dpct::group::store_algorithm S> bool test_group_store() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT 
  // dpct::group::load_algorithm::BLOCK_LOAD_STRIPED in its entirety as API
  // functions
  // Tests dpct::group::store_algorithm::BLOCK_STORE_DIRECT 
  // dpct::group::store_algorithm::BLOCK_STORE_STRIPED in its entirety as API
  // functions
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  int num_threads = 128;
  int items_per_thread = 4;
  if constexpr(S == dpct::group::store_algorithm::BLOCK_STORE_DIRECT) {
    for (int i = 0; i < 512; i++) {data[i] = i;}
  }
  else{
    for (int i = 0; i < num_threads; ++i) {
      for (int j = 0; j < items_per_thread; ++j) {
        data[i * items_per_thread + j] = j * num_threads + i;        
      }
      
    }
  }
  
  int data_out[512];
  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<int, 1> buffer_out(data_out, 512);
  
  q.submit([&](sycl::handler &h) {
    using group_store =
        dpct::group::workgroup_store<4, S, int, int *, sycl::nd_item<3>>;
    size_t temp_storage_size = group_store::get_local_memory_size(128);
    sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), h);
    sycl::accessor dacc_read(buffer, h, sycl::read_write);
    sycl::accessor dacc_write(buffer_out, h, sycl::read_write);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          int global_index =
              item.get_group(2) * item.get_local_range().get(2) +
              item.get_local_id(2); 
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            thread_data[i] = dacc_read[global_index * 4 + i];
          }
          
          auto *d_r =
              dacc_read.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *d_w =
              dacc_write.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          
          // Store thread_data of each work item from blocked arrangement
          group_store(tmp).store(item, d_r, thread_data);
          
          // Write thread_data of each work item at index to the global buffer
          global_index =
              item.get_group(2) * item.get_local_range().get(2) +
              item.get_local_id(2);  // Each thread_data has 4 elements
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            dacc_write[global_index * 4 + i] = thread_data[i];
          }
          
          });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer_out, sycl::read_write);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function<S>(ptr, "test_group_store");
}


bool test_store_subgroup_striped_standalone() {
  // Tests dpct::group::load_subgroup_striped as standalone method
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  for(int i=0;i<512;i++){data[i]=i;}
  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<uint32_t, 1> sg_sz_buf{sycl::range<1>(1)};
  int data_out[512];
  sycl::buffer<int, 1> buffer_out(data_out, 512);

  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc_read(buffer, h, sycl::read_write);
    sycl::accessor dacc_write(buffer_out, h, sycl::read_write);
    sycl::accessor sg_sz_dacc(sg_sz_buf, h, sycl::read_write);
    
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          int global_index =
              item.get_group(2) * item.get_local_range().get(2) +
              item.get_local_id(2); 
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            thread_data[i] = dacc_read[global_index * 4 + i];
          }
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
          global_index =
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



template <dpct::group::store_algorithm S> bool test_group_store_standalone() {
  // Tests dpct::group::load_algorithm::BLOCK_LOAD_DIRECT &
  // dpct::group::load_algorithm::BLOCK_LOAD_STRIPED as standalone methods
  // Tests dpct::group::store_algorithm::BLOCK_STORE_DIRECT &
  // dpct::group::store_algorithm::BLOCK_STORE_STRIPED as standalone methods
  sycl::queue q(dpct::get_default_queue());
  int data[512];
  int num_threads = 128;
  int items_per_thread = 4;
  if constexpr(S == dpct::group::store_algorithm::BLOCK_STORE_DIRECT) {
    for (int i = 0; i < 512; i++) {data[i] = i;}
  }
  else{
    for (int i = 0; i < num_threads; ++i) {
      for (int j = 0; j < items_per_thread; ++j) {
        data[i * items_per_thread + j] = j * num_threads + i;  
      }
      
    }
  }
  
  std::cout<<std::endl;
  int data_out[512];
  sycl::buffer<int, 1> buffer(data, 512);
  sycl::buffer<int, 1> buffer_out(data_out, 512);
  
  
  q.submit([&](sycl::handler &h) {
    sycl::accessor dacc_read(buffer, h, sycl::read_only);
    sycl::accessor dacc_write(buffer_out, h, sycl::read_write);
    //sycl::accessor data_accessor_read_write(buffer, h, sycl::read_write);
    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          int thread_data[4];
          int global_index =
              item.get_group(2) * item.get_local_range().get(2) +
              item.get_local_id(2); 
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            thread_data[i] = dacc_read[global_index * 4 + i];
          }
          
          //auto *d_r =
          //    dacc_read.get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *d_w =
              dacc_write.get_multi_ptr<sycl::access::decorated::yes>().get();
          // Store thread_data of each work item from blocked arrangement
          if (S == dpct::group::store_algorithm::BLOCK_STORE_DIRECT) {
            dpct::group::store_blocked<4, int>(item, d_w, thread_data);
          } else {
            dpct::group::store_striped<4, int>(item, d_w, thread_data);
          }
          // Write thread_data of each work item at index to the global buffer
          
          global_index =
              item.get_group(2) * item.get_local_range().get(2) +
              item.get_local_id(2);  // Each thread_data has 4 elements
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            dacc_write[global_index * 4 + i] = thread_data[i];
          }
        });
  });
  q.wait_and_throw();

  sycl::host_accessor data_accessor(buffer_out, sycl::read_write);
  const int *ptr = data_accessor.get_multi_ptr<sycl::access::decorated::yes>();
  return helper_validation_function<S>(ptr, "test_group_load_store");
}

int main() {
 
  return !(
      // Calls test_group_load with blocked and striped strategies , should pass
      // both results.
      test_group_store<dpct::group::store_algorithm::BLOCK_STORE_DIRECT>() &&
      test_group_store<dpct::group::store_algorithm::BLOCK_STORE_STRIPED>() &&
      // Calls test_load_subgroup_striped_standalone and should pass
      test_store_subgroup_striped_standalone() &
      // Calls test_group_load_standalone with blocked and striped strategies as
      // free functions, should pass both results.
      test_group_store_standalone<dpct::group::store_algorithm::BLOCK_STORE_DIRECT>() &&
      test_group_store_standalone<dpct::group::store_algorithm::BLOCK_STORE_STRIPED>());
}
