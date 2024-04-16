// ====------ onedpl_test_malloc_free.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/execution"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>
#include <vector>

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1 &&X, _T2 &&Y) {
  if (X != Y) {
    std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
    return 1;
  }
  return 0;
}

int test_passed(int failing_elems, std::string test_name) {
  if (failing_elems == 0) {
    std::cout << "PASS: " << test_name << std::endl;
    return 0;
  }
  return 1;
}

template <typename T, typename PolicyOrTag, typename TestVoidMalloc>
int test_malloc_free_on_device(sycl::queue q, PolicyOrTag policy_or_tag,
                               std::string test_name, std::size_t num_elements,
                               TestVoidMalloc) {
  using alloc_t = std::conditional_t<TestVoidMalloc::value, void, T>;
  int failed_tests = 0;
  dpct::tagged_pointer<alloc_t, dpct::device_sys_tag> ptr;
  if constexpr (TestVoidMalloc::value) {
    ptr = dpct::malloc(policy_or_tag, num_elements * sizeof(T));
  } else {
    ptr = dpct::malloc<T>(policy_or_tag, num_elements);
  }
  std::vector<T> num_data(num_elements);
  std::iota(num_data.begin(), num_data.end(), 0);
  std::vector<T> out_num_data(num_elements);
  q.submit([&](sycl::handler &h) {
     h.memcpy(ptr, num_data.data(), sizeof(T) * num_elements);
   })
      .wait();
  for (std::size_t i = 0; i != num_elements; ++i) {
    if constexpr (TestVoidMalloc::value)
      failed_tests += ASSERT_EQUAL(test_name, static_cast<T *>(ptr)[i],
                                   static_cast<T>(i));
    else
      failed_tests += ASSERT_EQUAL(test_name, ptr[i], static_cast<T>(i));
  }
  q.submit([&](sycl::handler &h) {
     h.memcpy(out_num_data.data(), ptr, num_elements * sizeof(T));
   })
      .wait();
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, out_num_data[i], static_cast<T>(i));

  test_passed(failed_tests, test_name);
  dpct::free(policy_or_tag, ptr);

  return failed_tests;
}

template <typename T, typename PolicyOrTag, typename TestVoidMalloc>
int test_malloc_free_on_host(PolicyOrTag policy_or_tag, std::string test_name,
                             std::size_t num_elements, TestVoidMalloc) {
  using alloc_t = std::conditional_t<TestVoidMalloc::value, void, T>;
  dpct::tagged_pointer<alloc_t, dpct::host_sys_tag> ptr;
  if constexpr (TestVoidMalloc::value) {
    ptr = dpct::malloc(policy_or_tag, num_elements * sizeof(T));
  } else {
    ptr = dpct::malloc<T>(policy_or_tag, num_elements);
  }
  int failed_tests = 0;
  std::vector<T> num_data(num_elements);
  std::iota(num_data.begin(), num_data.end(), 0);
  std::vector<T> out_num_data(num_elements);
  ::std::memcpy(ptr, num_data.data(), num_elements * sizeof(T));
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, static_cast<T *>(ptr)[i],
                                 static_cast<T>(i));
  ::std::memcpy(out_num_data.data(), ptr, num_elements * sizeof(T));
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, static_cast<T *>(ptr)[i],
                                 static_cast<T>(i));
  test_passed(failed_tests, test_name);
  dpct::free(policy_or_tag, ptr);

  return failed_tests;
}

int main() {
  int failed_tests = 0;

  // Test One, test normal calls for dpct::malloc and dpct::free by allocating a
  // certain number of bytes on device.
  {
    dpct::device_sys_tag device_exec;
    std::string test_name =
        "malloc and free byte array allocation - device memory with tag";
    failed_tests += test_malloc_free_on_device<double>(
        dpct::get_default_queue(), device_exec, test_name, 5, std::true_type{});
  }

  // Test Two, test normal calls for dpct::malloc and dpct::free by allocating a
  // certain number of bytes on host.
  {
    dpct::host_sys_tag host_exec;
    std::string test_name =
        "malloc and free byte array allocation - host memory with tag";
    failed_tests += test_malloc_free_on_host<double>(host_exec, test_name, 5,
                                                     std::true_type{});
  }

  // Test three, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device
  {
    dpct::device_sys_tag device_exec;
    std::string test_name =
        "malloc and free int64_t array allocation - device memory with tag";
    failed_tests += test_malloc_free_on_device<int64_t>(
        dpct::get_default_queue(), device_exec, test_name, 10,
        std::false_type{});
  }

  // Test four, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on host
  {
    dpct::host_sys_tag host_exec;
    std::string test_name =
        "malloc and free int64_t array allocation - host memory with tag";
    failed_tests += test_malloc_free_on_host<int64_t>(host_exec, test_name, 10,
                                                      std::false_type{});
  }

  // Test five, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device. oneDPL device policy passed as location
  {
    sycl::queue q = dpct::get_default_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    std::string test_name =
        "malloc and free int64_t array allocation - device memory with policy";
    failed_tests += test_malloc_free_on_device<int64_t>(q, policy, test_name,
                                                        10, std::false_type{});
  }

  // Test six, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device. oneDPL host policy passed as location
  {
    std::string test_name =
        "malloc and free int64_t array allocation - host memory with policy";
    failed_tests += test_malloc_free_on_host<int64_t>(
        oneapi::dpl::execution::seq, test_name, 10, std::false_type{});
  }

  // Test seven, ensure functionality of dpct::internal::malloc_base internal
  // utility.
  {
    std::size_t num_bytes = sizeof(int64_t) * 10;
    int64_t *ptr = static_cast<int64_t *>(
        dpct::internal::malloc_base(dpct::host_sys_tag{}, num_bytes));
    std::iota(ptr, ptr + 10, 0);
    std::string test_name = "internal::malloc_base utility with host tag";

    for (int i = 0; i < 10; ++i)
      failed_tests += ASSERT_EQUAL(test_name, ptr[i], i);
    std::free(ptr);
  }

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
