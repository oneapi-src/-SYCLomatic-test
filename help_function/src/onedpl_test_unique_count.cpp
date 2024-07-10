// ====------ onedpl_test_unique_count.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <sycl/sycl.hpp>

#include <iostream>
#include <type_traits>

template <typename String, typename _T1, typename _T2>
int
ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y)
{
    if (X != Y)
    {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    return 0;
}

int
test_passed(int failing_elems, std::string test_name)
{
    if (failing_elems == 0)
    {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

template <typename Buffer>
void
iota_buffer(Buffer& dst_buf, int start_index, int end_index, int offset)
{
    auto dst = dst_buf.get_host_access();
    for (int i = start_index; i != end_index; ++i)
    {
        dst[i] = i + offset;
    }
}

struct uint32_wrapper
{
    ::std::uint32_t val;
};

int
main()
{
    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";
    sycl::queue q(dpct::get_default_queue());
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    // 1. Test call with n runs of the form: [1, 2, 2, 3, 3, 3, ..., n]
    {
        test_name = "Testing 7 runs of form [1, 2, 2, 3, 3, 3, ..., 7]";
        std::size_t n = 7;
        sycl::buffer<std::uint32_t> src{sycl::range<1>((n * (n + 1)) / 2)};
        {
            auto acc = src.get_host_access();
            for (std::size_t i = 1; i <= n; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    acc[j + (i * (i - 1) / 2)] = i;
        }
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src));
        auto local_failures = ASSERT_EQUAL(test_name, res, n);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }
    // 2. Test case where each element is distinct run
    {
        test_name = "Testing 20 runs of form [1, 2, 3, ..., 19, 20]";
        std::size_t n = 20;
        sycl::buffer<std::uint32_t> src(n);
        iota_buffer(src, 0, n, 1);
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src));
        auto local_failures = ASSERT_EQUAL(test_name, res, n);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }
    // 3. Test 1 element case
    {
        test_name = "Testing 1 runs of form [30]";
        std::size_t n = 1;
        sycl::buffer<std::uint32_t> src(n);
        {
            auto acc = src.get_host_access();
            acc[0] = 30;
        }
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src));
        auto local_failures = ASSERT_EQUAL(test_name, res, n);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }
    // 4. Test 0 element case
    {
        test_name = "Testing 0 runs of form []";
        std::size_t n = 0;
        sycl::buffer<std::uint32_t> src(n);
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src));
        auto local_failures = ASSERT_EQUAL(test_name, res, 0);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }
    // 5. Test custom predicate
    {
        auto is_in_group_of_three = [](auto fst, auto snd) {
            using T = ::std::decay_t<decltype(fst)>;
            return static_cast<T>(fst / 3) == static_cast<T>(snd / 3);
        };
        test_name = "Testing custom predicate grouping runs of length 3";
        std::size_t n = 21;
        sycl::buffer<std::uint32_t> src(n);
        iota_buffer(src, 0, n, 0);
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src), is_in_group_of_three);
        auto local_failures = ASSERT_EQUAL(test_name, res, n / 3);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }
    // 6. Test custom predicate with custom type
    {
        auto is_equal_uint32_wrapper = [](uint32_wrapper fst, uint32_wrapper snd) { return fst.val == snd.val; };
        test_name = "Testing 16 runs of custom predicate and custom datatype";
        std::size_t n = 32;
        sycl::buffer<uint32_wrapper> src(n);
        {
            auto acc = src.get_host_access();
            for (int i = 0; i < n; ++i)
                acc[i] = uint32_wrapper{static_cast<uint32_t>(i / 2)};
        }
        auto res = dpct::unique_count(policy, dpl::begin(src), dpl::end(src), is_equal_uint32_wrapper);
        auto local_failures = ASSERT_EQUAL(test_name, res, n / 2);
        test_passed(local_failures, test_name);
        num_failing += local_failures;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0)
    {
        return 0;
    }
    return 1;
}
