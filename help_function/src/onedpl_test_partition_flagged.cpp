// ====------ onedpl_test_partition_flagged.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

#include <sycl/sycl.hpp>

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

template <typename ExecutionPolicy, typename InputIterator, typename FlagIterator, typename OutputIterator,
          typename CountIterator, typename ExpectedOutputIterator>
int
test_partition_flagged(const std::string& test_name, ExecutionPolicy&& policy, InputIterator input, FlagIterator flags,
                       OutputIterator output, CountIterator count, int num_elements, ExpectedOutputIterator expected,
                       int expected_partition_point)
{
    using OutputType = typename std::iterator_traits<OutputIterator>::value_type;
    using CountType = typename std::iterator_traits<CountIterator>::value_type;
    int num_failures = 0;
    bool rev_flag;
    for (int rev = 0; rev < 2; ++rev)
    {
        rev_flag = static_cast<bool>(rev);
        std::string reversed_msg = rev ? " w/ last partition reversal" : " w/o last partition reversal";
        CountType count_on_host;
        dpct::partition_flagged(policy, input, flags, output, count, num_elements, rev_flag);
        {
            auto count_buf = count.get_buffer();
            auto count_buf_acc = count_buf.get_host_access();
            count_on_host = count_buf_acc[0];
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " - output partition point", count_on_host,
                                         expected_partition_point);
        }
        {
            auto output_buf = output.get_buffer();
            auto output_buf_acc = output_buf.get_host_access();
            for (std::size_t i = 0; i < num_elements; ++i)
            {
                // expected is assumed to be loaded with the last partition reversed.
                std::size_t j =
                    (rev == 0 && i >= expected_partition_point) ? num_elements - i + expected_partition_point - 1 : i;
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " - output at idx " + std::to_string(i),
                                             output_buf_acc[i], expected[j]);
            }
        }
        // Flush output buffers between iterations.
        if (!rev_flag)
        {
            dpl::fill_n(policy, output, num_elements, OutputType{});
            dpl::fill_n(policy, count, 1, CountType{});
        }
    }
    return num_failures;
}

int
main()
{
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";
    sycl::queue q(dpct::get_default_queue());
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    // 1. Test 1 - Even / Odd
    {
        test_name = "Even / Odd partition - dpct::partition_flagged - int64_t";
        std::size_t num_elements = 8;
        // create buffer
        sycl::buffer<int64_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> flags_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> count_buf{sycl::range<1>(1)};
        {
            auto input_acc = input_buf.get_host_access();
            auto flags_acc = flags_buf.get_host_access();
            for (int i = 0; i < num_elements; ++i)
            {
                input_acc[i] = i;
                if (i % 2 == 0)
                    flags_acc[i] = 0;
                else
                    flags_acc[i] = 1;
            }
        }
        std::vector<int64_t> expected_output(num_elements);
        int expected_partition_point = 4;
        for (int i = 0; i < num_elements; ++i)
        {
            if (i < 4)
                expected_output[i] = 2 * i + 1; // fills first half with [1, 3, 5, 7]
            else
                expected_output[i] = (7 - i) * 2; // fills second half with [6, 4, 2, 0]
        }
        num_failing +=
            test_partition_flagged(test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(flags_buf),
                                   oneapi::dpl::begin(output_buf), oneapi::dpl::begin(count_buf), input_buf.size(),
                                   expected_output, expected_partition_point);
        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // 2. Test 2 - Already Partitioned
    {
        test_name = "Already partitioned - dpct::partition_flagged - int64_t";
        std::size_t num_elements = 30;
        // create buffer
        sycl::buffer<int64_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> flags_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> count_buf{sycl::range<1>(1)};
        {
            auto input_acc = input_buf.get_host_access();
            auto flags_acc = flags_buf.get_host_access();
            for (int i = 0; i != num_elements; ++i)
            {
                input_acc[i] = i;
                flags_acc[i] = 1;
            }
        }
        std::vector<int64_t> expected_output(num_elements);
        std::iota(expected_output.begin(), expected_output.end(), 0);
        int expected_partition_point = num_elements;
        num_failing +=
            test_partition_flagged(test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(flags_buf),
                                   oneapi::dpl::begin(output_buf), oneapi::dpl::begin(count_buf), input_buf.size(),
                                   expected_output, expected_partition_point);
        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 3 - Reverse Partitioned
    {
        test_name = "Reverse partition - dpct::partition_flagged - int64_t";
        std::size_t num_elements = 56;
        // create buffer
        sycl::buffer<int64_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> flags_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> count_buf{sycl::range<1>(1)};
        {
            auto input_acc = input_buf.get_host_access();
            auto flags_acc = flags_buf.get_host_access();
            for (int i = 0; i != num_elements; ++i)
            {
                input_acc[i] = i;
                flags_acc[i] = 0;
            }
        }
        std::vector<int64_t> expected_output(num_elements);
        std::iota(expected_output.rbegin(), expected_output.rend(), 0);
        int expected_partition_point = 0;
        num_failing +=
            test_partition_flagged(test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(flags_buf),
                                   oneapi::dpl::begin(output_buf), oneapi::dpl::begin(count_buf), input_buf.size(),
                                   expected_output, expected_partition_point);
        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 3 - Partition elements by whether they are divisible by 5
    {
        test_name = "Divisible by 5 - dpct::partition_flagged - uint32_t";
        std::size_t num_elements = 67;
        // create buffer
        sycl::buffer<std::uint32_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<std::uint32_t> flags_buf{sycl::range<1>(num_elements)};
        sycl::buffer<std::uint32_t> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<std::uint32_t> count_buf{sycl::range<1>(1)};
        {
            auto input_acc = input_buf.get_host_access();
            auto flags_acc = flags_buf.get_host_access();
            for (int i = 0; i != num_elements; ++i)
            {
                input_acc[i] = i;
                if (i % 5 == 0)
                    flags_acc[i] = 1;
                else
                    flags_acc[i] = 0;
            }
        }
        std::vector<int64_t> expected_output(num_elements);
        std::iota(expected_output.begin(), expected_output.end(), 0);
        auto not_div_5_beg =
            std::stable_partition(expected_output.begin(), expected_output.end(), [](auto e) { return e % 5 == 0; });
        std::reverse(not_div_5_beg, expected_output.end());
        int expected_partition_point = std::distance(expected_output.begin(), not_div_5_beg);
        num_failing +=
            test_partition_flagged(test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(flags_buf),
                                   oneapi::dpl::begin(output_buf), oneapi::dpl::begin(count_buf), input_buf.size(),
                                   expected_output, expected_partition_point);
        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0)
    {
        return 0;
    }
    return 1;
}
