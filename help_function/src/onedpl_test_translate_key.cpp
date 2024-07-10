// ====------ onedpl_test_translate_key.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

// TODO: If this test remains stable, then we may remove this macro entirely.
#define VERBOSE_DEBUG

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(_T1 &&X, _T2 &&Y, String msg) {
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

template <typename T_conv, typename T>
void expect_rvalues(T_conv &&_expected, T &&_actual, const char *err_string) {
  T expected = T(_expected);
  T actual = _actual;
  EXPECT_EQ(expected, actual, err_string);
}

int main() {

// Print sizeof types we test
#ifdef VERBOSE_DEBUG
    std::cout << "onedpl_test_translate_key - sizeof test types:" << std::endl;
    std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
    std::cout << "sizeof(float) = " << sizeof(float) << std::endl;
    std::cout << "sizeof(double) = " << sizeof(double) << std::endl;
#endif
  int test_suites_failed = 0;
  {
    ::std::string test_name = "translate int->uint32_t";
    auto trans_key = dpct::internal::translate_key<int, uint32_t>(0, 32);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0x7fffff9cU, trans_key(-100), test_name);
    tests_failed += ASSERT_EQUAL(0x80000064U, trans_key(100), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }

  {
    ::std::string test_name = "translate int->uint16_t";
    auto trans_key = dpct::internal::translate_key<int, uint16_t>(0, 16);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0xff9cU, trans_key(-100), test_name);
    tests_failed += ASSERT_EQUAL(0x0064U, trans_key(100), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "translate int->uint32_t";
    auto trans_key = dpct::internal::translate_key<int, uint16_t>(16, 32);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0x7fffU, trans_key(-100), test_name);
    tests_failed += ASSERT_EQUAL(0x8000U, trans_key(100), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }

  {
    ::std::string test_name = "translate int->uint16_t bits:[8-24)";
    auto trans_key = dpct::internal::translate_key<int, uint16_t>(8, 24);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0xffffU, trans_key(-100), test_name);
    tests_failed += ASSERT_EQUAL(0x0000U, trans_key(100), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "translate int->uint8_t bits:[8-13)";
    auto trans_key = dpct::internal::translate_key<int, uint8_t>(8, 13);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0x1fU, trans_key(-100), test_name);
    tests_failed += ASSERT_EQUAL(0x00U, trans_key(100), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }

  {
    ::std::string test_name = "translate float->uint32_t bits:[0-32)";
    auto trans_key = dpct::internal::translate_key<float, uint32_t>(0, 32);
    int tests_failed = 0;
    tests_failed += ASSERT_EQUAL(0x3c55e20bU, trans_key(-340.234f), test_name);
    tests_failed += ASSERT_EQUAL(0x80000000U, trans_key(-0.0f), test_name);
    tests_failed += ASSERT_EQUAL(0x80000000U, trans_key(0.0f), test_name);
    tests_failed += ASSERT_EQUAL(0xc6408c00U, trans_key(12323.0f), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "translate double->uint64_t bits:[0-64)";
    auto trans_key = dpct::internal::translate_key<double, uint64_t>(0, 64);
    int tests_failed = 0;
    tests_failed +=
        ASSERT_EQUAL(0x3f8abc4189374bc6UL, trans_key(-340.234), test_name);
    tests_failed +=
        ASSERT_EQUAL(0x8000000000000000UL, trans_key(-0.0), test_name);
    tests_failed +=
        ASSERT_EQUAL(0x8000000000000000UL, trans_key(0.0), test_name);
    tests_failed +=
        ASSERT_EQUAL(0xc0c8118000000000UL, trans_key(12323.0), test_name);
    test_suites_failed += test_passed(tests_failed, test_name);
  }

  std::cout << std::endl
            << test_suites_failed << " failing test(s) detected." << std::endl;
  if (test_suites_failed == 0) {
    return 0;
  }
  return 1;
}
