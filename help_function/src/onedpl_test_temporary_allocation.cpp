// ====------ onedpl_test_temporary_allocation.cpp---------- -*- C++ -*
// ----===////
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

template <typename T, typename PolicyOrTag>
int test_temporary_allocation_on_device(sycl::queue q,
                                        PolicyOrTag policy_or_tag,
                                        std::string test_name,
                                        std::size_t num_elements) {
  int failed_tests = 0;
  // TODO: Use structured bindings when we switch to C++20 and can capture the
  // variables in lambda capture clause.
  auto ret_tup = dpct::get_temporary_allocation<T>(policy_or_tag, num_elements);
  auto ptr = ret_tup.first;
  auto num_allocated = ret_tup.second;
  std::vector<T> num_data(num_elements);
  std::iota(num_data.begin(), num_data.end(), 0);
  std::vector<T> out_num_data(num_elements);
  q.submit([&](sycl::handler &h) {
     h.memcpy(ptr, num_data.data(), sizeof(T) * num_elements);
   })
      .wait();
  for (std::size_t i = 0; i != num_elements; ++i) {
    failed_tests += ASSERT_EQUAL(test_name, ptr[i], static_cast<T>(i));
  }
  q.submit([&](sycl::handler &h) {
     h.memcpy(out_num_data.data(), ptr, num_elements * sizeof(T));
   })
      .wait();
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, out_num_data[i], static_cast<T>(i));

  failed_tests += (num_allocated != num_elements);
  test_passed(failed_tests, test_name);
  dpct::release_temporary_allocation(policy_or_tag, ptr);

  return failed_tests;
}

template <typename T, typename PolicyOrTag>
int test_temporary_allocation_on_host(PolicyOrTag policy_or_tag,
                                      std::string test_name,
                                      std::size_t num_elements) {
  int failed_tests = 0;
  // TODO: Use structured bindings when we switch to C++20 and can capture the
  // variables in lambda capture clause.
  auto ret_tup = dpct::get_temporary_allocation<T>(policy_or_tag, num_elements);
  auto ptr = ret_tup.first;
  auto num_allocated = ret_tup.second;
  std::vector<T> num_data(num_elements);
  std::iota(num_data.begin(), num_data.end(), 0);
  std::vector<T> out_num_data(num_elements);
  ::std::memcpy(ptr, num_data.data(), num_elements * sizeof(T));
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, ptr[i], static_cast<T>(i));
  ::std::memcpy(out_num_data.data(), ptr, num_elements * sizeof(T));
  for (std::size_t i = 0; i != num_elements; ++i)
    failed_tests += ASSERT_EQUAL(test_name, ptr[i], static_cast<T>(i));

  failed_tests += (num_allocated != num_elements);
  test_passed(failed_tests, test_name);
  dpct::release_temporary_allocation(policy_or_tag, ptr);

  return failed_tests;
}

int main() {
  // used to detect failures
  int failed_tests = 0;

  // Test One, temporary allocation with device tag.
  {
    dpct::device_sys_tag device_sys;
    std::string test_name =
        "get and release temporary allocation - device memory with tag";
    failed_tests += test_temporary_allocation_on_device<int64_t>(
        dpct::get_default_queue(), device_sys, test_name, 10);
  }

  // Test Two, temporary allocation with host tag.
  {
    dpct::host_sys_tag host_sys;
    std::string test_name =
        "get and release temporary allocation - host memory with tag";
    failed_tests +=
        test_temporary_allocation_on_host<int64_t>(host_sys, test_name, 10);
  }

  // Test Three, temporary allocation with device policy.
  {
    sycl::queue q = dpct::get_default_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    std::string test_name =
        "get and release temporary allocation - device memory with policy";
    failed_tests +=
        test_temporary_allocation_on_device<int64_t>(q, policy, test_name, 10);
  }

  // Test Four, temporary allocation with host policy.
  {
    std::string test_name =
        "get and release temporary allocation - host memory with policy";
    failed_tests += test_temporary_allocation_on_host<int64_t>(
        oneapi::dpl::execution::par_unseq, test_name, 10);
  }

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
