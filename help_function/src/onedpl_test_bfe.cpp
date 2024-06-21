// ====------ onedpl_test_bfe.cpp------------------------ -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <oneapi/dpl/execution>
#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"
#include "dpct/dpl_extras/functional.h"
#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
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

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    {
        std::uint8_t orig = 240;
        std::string test_name = "dpct::bfe test std::uint8_t";
        //casting to uint32_t so that cout does not confuse with a char
        failed_tests += ASSERT_EQUAL(test_name, (::std::uint32_t)dpct::bfe(orig, 0, 4), 0);
        failed_tests += ASSERT_EQUAL(test_name, (::std::uint32_t)dpct::bfe(orig, 4, 4), 15);
        
        failed_tests += ASSERT_EQUAL(test_name, (::std::uint32_t)dpct::bfe(orig, 1, 6), 56);
        failed_tests += ASSERT_EQUAL(test_name, (::std::uint32_t)dpct::bfe(orig, 5, 3), 7);
        test_passed(failed_tests, test_name);
    }

    {
        std::uint16_t orig = 54444;
        std::string test_name = "dpct::bfe test std::uint16_t";
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 0, 8), 172);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 8, 8), 212);
        
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 3, 12), 2709);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 1, 3), 6);
        test_passed(failed_tests, test_name);
    }

    {
        std::uint32_t orig = 123023024;
        std::string test_name = "dpct::bfe test std::uint32_t";
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 0, 8), 176);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 8, 8), 46);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 16, 8), 85);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 24, 8), 7);
        
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 28, 4), 0);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 13, 10), 681);
        test_passed(failed_tests, test_name);
    }

    {
        std::uint64_t orig = 128765854343023024ULL;
        std::string test_name = "dpct::bfe test std::uint64_t";
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 0, 16), 51632);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 16, 16), 48976);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 32, 16), 30684);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 48, 16), 457);
        
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 55, 4), 3);
        failed_tests += ASSERT_EQUAL(test_name, dpct::bfe(orig, 13, 13), 6790);
        test_passed(failed_tests, test_name);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
