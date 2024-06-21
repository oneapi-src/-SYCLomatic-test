// ====------ onedpl_test_partition_if.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>

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

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename CountIterator,
          typename ExpectedOutputIterator, typename UnaryPredicate>
int
test_partition_if_1output(const std::string& test_name, ExecutionPolicy&& policy, InputIterator input,
                          OutputIterator output, CountIterator count, int num_elements, UnaryPredicate unary_pred,
                          ExpectedOutputIterator expected, int expected_partition_point)
{
    using OutputType = typename std::iterator_traits<OutputIterator>::value_type;
    using CountType = typename std::iterator_traits<CountIterator>::value_type;
    int num_failures = 0;
    bool rev_flag;
    for (int rev = 0; rev < 2; ++rev)
    {
        CountType count_on_host;
        rev_flag = static_cast<bool>(rev);
        dpct::partition_if(policy, input, output, count, num_elements, unary_pred, rev_flag);
        {
            auto count_buf = count.get_buffer();
            auto count_buf_acc = count_buf.get_host_access();
            count_on_host = count_buf_acc[0];
            num_failures +=
                ASSERT_EQUAL(test_name + " - output partition point", count_on_host, expected_partition_point);
        }
        {
            auto output_buf = output.get_buffer();
            auto output_buf_acc = output_buf.get_host_access();
            std::string reversed_msg = rev ? " w/ last partition reversal" : " w/o last partition reversal";
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

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator1, typename OutputIterator2,
          typename OutputIterator3, typename CountIterator, typename UnaryPredicate1, typename UnaryPredicate2,
          typename ExpectedOutputIterator1, typename ExpectedOutputIterator2, typename ExpectedOutputIterator3>
int
test_partition_if_3outputs(const std::string& test_name, ExecutionPolicy&& policy, InputIterator input,
                           OutputIterator1 output1, OutputIterator2 output2, OutputIterator3 output3,
                           CountIterator counts, int num_elements, UnaryPredicate1 pred1, UnaryPredicate2 pred2,
                           ExpectedOutputIterator1 expected1, int expected_elements1, ExpectedOutputIterator2 expected2,
                           int expected_elements2, ExpectedOutputIterator3 expected3, int expected_elements3)
{
    using CountType = typename std::iterator_traits<CountIterator>::value_type;
    using OutputType1 = typename std::iterator_traits<OutputIterator1>::value_type;
    using OutputType2 = typename std::iterator_traits<OutputIterator2>::value_type;
    using OutputType3 = typename std::iterator_traits<OutputIterator3>::value_type;
    int num_failures = 0;
    bool rev_flag;
    for (int rev = 0; rev < 2; ++rev)
    {
        rev_flag = static_cast<bool>(rev);
        int host_count_output1 = 0, host_count_output2 = 0, host_count_output3 = 0;
        std::string reversed_msg = rev ? " w/ last partition reversal" : " w/o last partition reversal";
        dpct::partition_if(policy, input, output1, output2, output3, counts, num_elements, pred1, pred2, rev_flag);
        {
            auto counts_buf = counts.get_buffer();
            auto counts_acc = counts_buf.get_host_access();
            host_count_output1 = counts_acc[0];
            host_count_output2 = counts_acc[1];
            host_count_output3 = counts_acc[2];
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 1 ", host_count_output1,
                                         expected_elements1);
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 2 ", host_count_output2,
                                         expected_elements2);
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 3 ", host_count_output3,
                                         expected_elements3);
        }
        {
            auto output1_buf = output1.get_buffer();
            auto output2_buf = output2.get_buffer();
            auto output3_buf = output3.get_buffer();
            auto output1_acc = output1_buf.get_host_access();
            auto output2_acc = output2_buf.get_host_access();
            auto output3_acc = output3_buf.get_host_access();

            for (std::size_t i = 0; i < host_count_output1; ++i)
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output1 at idx " + std::to_string(i),
                                             output1_acc[i], expected1[i]);
            for (std::size_t i = 0; i < host_count_output2; ++i)
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output2 at idx " + std::to_string(i),
                                             output2_acc[i], expected2[i]);
            for (std::size_t i = 0; i < host_count_output3; ++i)
            {
                // expected is assumed to be loaded with the last partition reversed.
                std::size_t j = rev_flag ? i : host_count_output3 - i - 1;
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output3 at idx " + std::to_string(i),
                                             output3_acc[i], expected3[j]);
            }
        }
        // Flush output buffers between iterations.
        if (!rev_flag)
        {
            dpl::fill_n(policy, output1, num_elements, OutputType1{});
            dpl::fill_n(policy, output2, num_elements, OutputType2{});
            dpl::fill_n(policy, output3, num_elements, OutputType3{});
            dpl::fill_n(policy, counts, 3, CountType{});
        }
    }
    return num_failures;
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator1, typename OutputIterator2,
          typename OutputIterator3, typename CountIterator, typename UnaryPredicate1, typename UnaryPredicate2,
          typename ExpectedOutputIterator1, typename ExpectedOutputIterator2, typename ExpectedOutputIterator3>
int
test_partition_if_3outputs_rev_itr(const std::string& test_name, ExecutionPolicy&& policy, InputIterator input,
                                   OutputIterator1 output1, OutputIterator2 output2,
                                   std::reverse_iterator<OutputIterator3> output3, CountIterator counts,
                                   int num_elements, UnaryPredicate1 pred1, UnaryPredicate2 pred2,
                                   ExpectedOutputIterator1 expected1, int expected_elements1,
                                   ExpectedOutputIterator2 expected2, int expected_elements2,
                                   ExpectedOutputIterator3 expected3, int expected_elements3)
{
    using CountType = typename std::iterator_traits<CountIterator>::value_type;
    using OutputType1 = typename std::iterator_traits<OutputIterator1>::value_type;
    using OutputType2 = typename std::iterator_traits<OutputIterator2>::value_type;
    using OutputType3 = typename std::iterator_traits<OutputIterator3>::value_type;
    int num_failures = 0;
    bool rev_flag;
    for (int rev = 0; rev < 2; ++rev)
    {
        rev_flag = static_cast<bool>(rev);
        CountType host_count_output1 = 0, host_count_output2 = 0, host_count_output3 = 0;
        std::string reversed_msg = rev ? " w/ last partition reversal" : " w/o last partition reversal";
        dpct::partition_if(policy, input, output1, output2, output3, counts, num_elements, pred1, pred2, rev_flag);
        {
            std::vector<CountType> host_counts(3);
            policy.queue().copy(counts, host_counts.data(), 3).wait();
            host_count_output1 = host_counts[0], host_count_output2 = host_counts[1],
            host_count_output3 = host_counts[2];
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 1 ", host_count_output1,
                                         expected_elements1);
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 2 ", host_count_output2,
                                         expected_elements2);
            num_failures += ASSERT_EQUAL(test_name + reversed_msg + " partitioned count output 3 ", host_count_output3,
                                         expected_elements3);
        }
        {
            std::vector<OutputType1> host_output1(host_count_output1);
            std::vector<OutputType2> host_output2(host_count_output2);
            std::vector<OutputType3> host_output3(host_count_output3);
            // Since output3 is a reverse iterator, the values are loaded from the end.
            auto output3_offset = output3.base() - host_count_output3;
            policy.queue().copy(output1, host_output1.data(), host_count_output1);
            policy.queue().copy(output2, host_output2.data(), host_count_output2);
            policy.queue().copy(output3_offset, host_output3.data(), host_count_output3);
            policy.queue().wait();

            for (std::size_t i = 0; i < host_count_output1; ++i)
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output1 at idx " + std::to_string(i),
                                             host_output1[i], expected1[i]);
            for (std::size_t i = 0; i < host_count_output2; ++i)
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output2 at idx " + std::to_string(i),
                                             host_output2[i], expected2[i]);
            for (std::size_t i = 0; i < host_count_output3; ++i)
            {
                // expected is assumed to be loaded with the last partition reversed.
                std::size_t j = rev_flag ? i : host_count_output3 - i - 1;
                num_failures += ASSERT_EQUAL(test_name + reversed_msg + " output3 at idx " + std::to_string(i),
                                             host_output3[i], expected3[j]);
            }
        }
        // Flush output buffers between iterations.
        if (!rev_flag)
        {
            dpl::fill_n(policy, output1, num_elements, OutputType1{});
            dpl::fill_n(policy, output2, num_elements, OutputType2{});
            dpl::fill_n(policy, output3, num_elements, OutputType3{});
            dpl::fill_n(policy, counts, 3, CountType{});
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
    // Test 1. Even / Odd partition to single output buffer
    {
        test_name = "Even / Odd partition - dpct::partition_if with single output "
                    "- int64_t";
        std::size_t num_elements = 8;
        // create buffer
        sycl::buffer<int64_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> num_buf_out{sycl::range<1>(num_elements)};

        std::vector<int64_t> expected_output(num_elements);
        int expected_partition_point = 4;
        for (int i = 0; i < num_elements; ++i)
        {
            if (i < 4)
                expected_output[i] = 2 * i + 1; // fills first half with [1, 3, 5, 7]
            else
                expected_output[i] = (7 - i) * 2; // fills second half with [6, 4, 2, 0]
        }
        auto is_odd = [](auto e) { return e % 2; };
        iota_buffer(input_buf, 0, input_buf.size(), 0);
        num_failing = test_partition_if_1output(test_name, policy, oneapi::dpl::begin(input_buf),
                                                oneapi::dpl::begin(output_buf), oneapi::dpl::begin(num_buf_out), 8,
                                                is_odd, expected_output.begin(), expected_partition_point);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 2. Reverse the output
    {
        auto is_pos = [](auto n) { return n > 0; };
        test_name = "Reverse the input buffer - dpct::partition_if with single output "
                    "- float";
        std::size_t num_elements = 27;
        // create buffer
        sycl::buffer<float> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<float> output_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> num_buf_out{sycl::range<1>(1)};

        std::vector<float> expected_output(num_elements);
        std::iota(expected_output.rbegin(), expected_output.rend(), -30.);
        int expected_partition_point = 0;
        iota_buffer(input_buf, 0, input_buf.size(), -30.);
        num_failing = test_partition_if_1output(
            test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(output_buf),
            oneapi::dpl::begin(num_buf_out), num_elements, is_pos, expected_output.begin(), expected_partition_point);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 3. Positive, negative, and zero partition
    {
        test_name = "Pos / Neg / Zero partition - dpct::partition_if with three "
                    "outputs - int64_t";
        std::size_t num_elements = 11;
        auto is_pos = [](auto n) { return n > 0; };
        auto is_neg = [](auto n) { return n < 0; };

        // create buffer
        sycl::buffer<int64_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> is_positive_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> is_negative_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> is_neither_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int64_t> counts{sycl::range<1>(3)};

        std::vector<int64_t> expected1(5);
        std::vector<int64_t> expected2(5);
        std::vector<int64_t> expected3(1);
        std::iota(expected1.begin(), expected1.end(), 1);
        std::iota(expected2.begin(), expected2.end(), -5);
        {
            auto is_neither_acc = is_neither_buf.get_host_access();
            is_neither_acc[0] = 0xff; // Store some garbage value since the output should be zero.
            iota_buffer(input_buf, 0, input_buf.size(), -5);
        }
        num_failing += test_partition_if_3outputs(
            test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(is_positive_buf),
            oneapi::dpl::begin(is_negative_buf), oneapi::dpl::begin(is_neither_buf), oneapi::dpl::begin(counts),
            num_elements, is_pos, is_neg, expected1.begin(), 5, expected2.begin(), 5, expected3.begin(), 1);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 4. Leave first output empty
    {
        test_name = "Leave the first output bucket empty - dpct::partition_if with three "
                    "outputs - int32_t";
        auto is_lt_0 = [](auto n) { return n < 0; };
        auto is_gt_100 = [](auto n) { return n > 100; };

        int num_elements = 20;
        // create buffer
        sycl::buffer<int32_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> is_less_than_0_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> is_greater_than_100_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> is_neither_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> counts{sycl::range<1>(3)};

        std::vector<int32_t> expected1(0);
        std::vector<int64_t> expected2(num_elements / 2);
        std::vector<int64_t> expected3(num_elements / 2);
        {
            auto input_acc = input_buf.get_host_access();
            // [0, 101, 2, 103, ...]
            for (int i = 0; i < num_elements; ++i)
            {
                if (i % 2 == 0)
                {
                    input_acc[i] = i;
                    expected3[i / 2] = num_elements - 2 - i;
                }
                else
                {
                    input_acc[i] = i + 100;
                    expected2[i / 2] = i + 100;
                }
            }
        }
        num_failing += test_partition_if_3outputs(
            test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(is_less_than_0_buf),
            oneapi::dpl::begin(is_greater_than_100_buf), oneapi::dpl::begin(is_neither_buf), oneapi::dpl::begin(counts),
            num_elements, is_lt_0, is_gt_100, expected1.begin(), 0, expected2.begin(), num_elements / 2,
            expected3.begin(), num_elements / 2);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 5. Leave last output empty (equivalent semantics to std::partition_copy)
    {
        test_name = "Leave the last output bucket empty - dpct::partition_if with three "
                    "outputs - int32_t";

        auto divisible_by_6 = [](auto e) { return e % 6 == 0; };
        auto not_divisible_by_6 = [](auto e) { return e % 6 != 0; };
        int num_elements = 40;
        // create buffer
        sycl::buffer<int32_t> input_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> is_divisible_by_6_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> not_divisible_by_6_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> is_neither_buf{sycl::range<1>(num_elements)};
        sycl::buffer<int32_t> counts{sycl::range<1>(3)};

        std::vector<int32_t> expected1(num_elements / 6);
        std::vector<int32_t> expected2(num_elements - num_elements / 6);
        std::vector<int32_t> expected3(0);
        {
            auto input_acc = input_buf.get_host_access();
            int count1 = 0, count2 = 0;
            for (int i = 1; i < num_elements + 1; ++i)
            {
                input_acc[i - 1] = i;
                if (i % 6 == 0)
                {
                    expected1[count1++] = i;
                }
                else
                {
                    expected2[count2++] = i;
                }
            }
        }
        num_failing += test_partition_if_3outputs(
            test_name, policy, oneapi::dpl::begin(input_buf), oneapi::dpl::begin(is_divisible_by_6_buf),
            oneapi::dpl::begin(not_divisible_by_6_buf), oneapi::dpl::begin(is_neither_buf), oneapi::dpl::begin(counts),
            num_elements, divisible_by_6, not_divisible_by_6, expected1.begin(), num_elements / 6, expected2.begin(),
            num_elements - num_elements / 6, expected3.begin(), 0);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
    // Test 6. 3 Output partition where second and third outputs are a combined allocation. Third output
    // is a std::reverse_iterator.
    {
        test_name = "Partition numbers into three ranges - dpct::partition_if with three "
                    "outputs - int32_t usm - combined allocation for second and third outputs";
        int num_elements = 30;
        auto is_greater_than_20 = [](auto e) { return e > 20; };
        auto is_less_than_10 = [](auto e) { return e < 10; };

        auto input_ptr = sycl::malloc_device<int32_t>(num_elements, q);
        auto greater_than_20_ptr = sycl::malloc_device<int32_t>(num_elements, q);
        auto less_than_10_and_unselected_ptr = sycl::malloc_device<int32_t>(num_elements, q);
        auto count_ptr = sycl::malloc_device<int32_t>(3, q);
        dpct::iota(policy, input_ptr, input_ptr + num_elements, 0);

        std::vector<int32_t> expected1(num_elements / 3 - 1);
        std::vector<int32_t> expected2(num_elements / 3);
        std::vector<int32_t> rev_expected3(num_elements / 3 + 1);
        std::iota(expected1.begin(), expected1.end(), 21);
        std::iota(expected2.begin(), expected2.end(), 0);
        std::iota(rev_expected3.begin(), rev_expected3.end(), 10);

        auto rev_unselected_beg = std::make_reverse_iterator(less_than_10_and_unselected_ptr + num_elements);
        num_failing += test_partition_if_3outputs_rev_itr(
            test_name, policy, input_ptr, greater_than_20_ptr, less_than_10_and_unselected_ptr, rev_unselected_beg,
            count_ptr, num_elements, is_greater_than_20, is_less_than_10, expected1.begin(), expected1.size(),
            expected2.begin(), expected2.size(), rev_expected3.begin(), rev_expected3.size());

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;

        sycl::free(input_ptr, q);
        sycl::free(greater_than_20_ptr, q);
        sycl::free(less_than_10_and_unselected_ptr, q);
        sycl::free(count_ptr, q);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0)
    {
        return 0;
    }
    return 1;
}
