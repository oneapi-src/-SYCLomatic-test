// ====------ onedpl_test_device_ptr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y, bool skip_pass_msg = false) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else if (!skip_pass_msg){
        std::cout << "PASS: " << msg << std::endl;
    }
    return 0;
}

template <typename String, typename _T1, typename _T2>
int
ASSERT_EQUAL_N(String msg, _T1&& X, _T2&& Y, ::std::size_t n)
{
    int failed_tests = 0;
    for (size_t i = 0; i < n; i++)
    {
        failed_tests += ASSERT_EQUAL(msg, *X, *Y, true);
        X++;
        Y++;
    }
    if (failed_tests == 0)
    {
        std::cout << "PASS: " << msg << std::endl;
    }
    return failed_tests;
}

int test_device_ptr_manipulation(void)
{
    int failing_tests = 0;
    typedef int T;

#ifdef DPCT_USM_LEVEL_NONE
    sycl::buffer<T, 1> data(sycl::range<1>(5));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 5);
#else
    dpct::device_pointer<T> data(5 * sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 5);
#endif

    failing_tests += ASSERT_EQUAL("device_ptr test 1", end - begin, 5);

    begin++;
    begin--;

    failing_tests += ASSERT_EQUAL("device_ptr test 2", end - begin, 5);

    begin += 1;
    begin -= 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 3", end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 4", end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 5", end - begin, 5);

    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 6", end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 7", end - begin, 5);

    begin = begin + (dpct::device_pointer<T>::difference_type) 1;
    begin = begin - (dpct::device_pointer<T>::difference_type) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 8", end - begin, 5);

    dpct::device_iterator<T> iter(begin);
    *begin = 3;
    failing_tests += ASSERT_EQUAL("device_ptr test 9", *begin, 3);
    failing_tests += ASSERT_EQUAL("device_ptr test 10", *iter, 3);

    return failing_tests;
}

int test_device_ptr_iteration(void)
{
    int failing_tests = 0;
    typedef size_t T;

#ifdef DPCT_USM_LEVEL_NONE
    sycl::buffer<T, 1> data(sycl::range<1>(1024));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 1024);
#else
    dpct::device_pointer<T> data(1024*sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 1024);
#endif
    auto policy = oneapi::dpl::execution::make_device_policy(dpct::get_default_queue());

    std::fill(policy, begin, end, 99);
    T result = oneapi::dpl::transform_reduce(policy, begin, end, static_cast<T>(0), std::plus<T>(), oneapi::dpl::identity());
    failing_tests += ASSERT_EQUAL("device_ptr in transform_reduce", result, 1024*99);
    return failing_tests;
}

int
test_permutation_iterator()
{
    int failing_tests = 0;
    typedef size_t T;
#ifdef DPCT_USM_LEVEL_NONE
    sycl::buffer<T, 1> data(sycl::range<1>(1024));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 1024);

    sycl::buffer<T, 1> data_res(sycl::range<1>(1024));
    dpct::device_pointer<T> begin_res(data_res, 0);
    dpct::device_pointer<T> end_res(data_res, 1024);
#else
    dpct::device_pointer<T> data(1024*sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 1024);

    dpct::device_pointer<T> data_res(1024*sizeof(T));
    dpct::device_pointer<T> begin_res(data_res);
    dpct::device_pointer<T> end_res(data_res + 1024);
#endif
    auto policy = oneapi::dpl::execution::make_device_policy(dpct::get_default_queue());
    std::fill(begin, end, T(1));
    std::fill(begin_res, end_res, T(99));
    auto perm = oneapi::dpl::make_permutation_iterator(begin, oneapi::dpl::counting_iterator(0));
    std::copy(policy, perm, perm + 1024, begin_res);
    return ASSERT_EQUAL_N("device_ptr in permutation_iterator", begin_res, dpct::make_constant_iterator(T(1)), 1024);
}

int main() {
    int failed_tests = 0;
    failed_tests += test_device_ptr_manipulation();
    failed_tests += test_device_ptr_iteration();
    failed_tests += test_permutation_iterator();

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
