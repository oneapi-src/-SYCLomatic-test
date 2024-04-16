// ====------ onedpl_test_uninitialized_fill.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/memory"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>
#include <iomanip>

#include <sycl/sycl.hpp>

// TODO: If this test remains stable, then we may remove this macro entirely.
#define VERBOSE_DEBUG

struct Int
{
    Int(int x) : val(x) {}
    int val;
};

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
    std::string test_name = "";

    // test 1/2

    const int N = 16;

    Int obj(10);
    dpct::device_pointer<Int> array = dpct::malloc_device<Int>(N * sizeof(Int));

    // call algorithm
    std::uninitialized_fill(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), array, array + N, obj);

    // copy back to host
    std::vector<Int> host_array(N, Int(0));
    dpct::get_default_queue().submit([&](sycl::handler& h) {
        h.memcpy(host_array.data(), array, N * sizeof(Int));
    });

    dpct::get_default_queue().wait();

    test_name = "uninitialized_fill with USM allocation";
    for (int i = 0; i != N; ++i) {
        num_failing += ASSERT_EQUAL(test_name, host_array[i].val, 10);
    }

    failed_tests += test_passed(num_failing, test_name);

    // test 2/2

    Int obj2(4);
    Int obj3(12);
    dpct::device_pointer<Int> array2 = dpct::malloc_device<Int>(N * sizeof(Int));

    // call algorithm
    std::uninitialized_fill(oneapi::dpl::execution::make_device_policy<>(dpct::get_default_queue()), array2, array2 + 8, obj2);
    std::uninitialized_fill(oneapi::dpl::execution::make_device_policy<>(dpct::get_default_queue()), array2 + 8, array2 + N, obj3);

    dpct::get_default_queue().wait();

    // copy back to host
    std::fill(host_array.begin(), host_array.end(), Int(0));
    dpct::get_default_queue().submit([&](sycl::handler& h) {
        h.memcpy(host_array.data(), array2, N * sizeof(Int));
    });

    dpct::get_default_queue().wait();

    test_name = "uninitialized_fill with USM allocation 2";
    int test2_num_failing = 0;
    for (int i = 0; i != N; ++i) {
        if (i < 8)
            test2_num_failing += ASSERT_EQUAL(test_name, host_array[i].val, obj2.val);
        else
            test2_num_failing += ASSERT_EQUAL(test_name, host_array[i].val, obj3.val);
    }
    num_failing += test2_num_failing;
#ifdef VERBOSE_DEBUG
    if (test2_num_failing > 0) {
        std::cout << test_name << " Expected: ";
        for (int i = 0; i != N; ++i) {
            if (i < 8)
                std::cout << obj2.val << " ";
            else
                std::cout << obj3.val << " ";
        }
        std::cout << std::endl << test_name << " Actual: ";
        for (int i = 0; i != N; ++i) {
                std::cout << host_array[i].val << " ";
        }
        std::cout << std::endl;
    }
#endif

    failed_tests += test_passed(num_failing, test_name);

    dpct::get_default_queue().wait();

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
