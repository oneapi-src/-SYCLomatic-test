// ====------ onedpl_test_reduce.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/algorithm"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <limits>



template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

template<typename Buffer> void fill_buffer(Buffer& src_buf, int start_index, int end_index, uint64_t value) {
    auto src = src_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        src[i] = value;
    }
}

template<typename Buffer> void iota_buffer(Buffer& dst_buf, int start_index, int end_index) {
    auto dst = dst_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        dst[i] = i;
    }
}

struct oppositeFunctor {
    std::tuple<bool, bool, bool> operator()(const std::tuple<bool, bool, bool> &t) const {
        return std::tuple<bool, bool, bool>
        (
            !std::get<0>(t),
            !std::get<1>(t),
            !std::get<2>(t)
        );
    }
};

struct andBooleanFunctor {
    std::tuple<bool, bool, bool> operator()(const std::tuple<bool, bool, bool>& t1, const std::tuple<bool, bool, bool>& t2) const {
        return std::tuple<bool, bool, bool>
        (
            std::get<0>(t1) && std::get<0>(t2),
            std::get<1>(t1) && std::get<1>(t2),
            std::get<2>(t1) && std::get<2>(t2)
        );
    }
};

struct square {
    int operator()(int x) const {
        return x*x;
    }
};

struct multiply {
    template <typename T>
    typename std::remove_reference<typename std::tuple_element<0, typename std::decay<T>::type>::type>::type
    operator()(T&& t) const {
        return std::get<0>(t) * std::get<1>(t);
    }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // First Group: Testing regular calls to std::reduce
    {
        // test 1/2

        // create buffer
        sycl::buffer<uint64_t, 1> src_buf { sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);

        iota_buffer(src_buf, 0, 8);

        // call algorithm
        auto result = std::reduce(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, 0, std::plus<int>());

        test_name = "Regular call to std::reduce 1/2";
        failed_tests += ASSERT_EQUAL(test_name, result, 6);

        // test 2/2

        // call algorithm
        result = std::reduce(oneapi::dpl::execution::dpcpp_default, src_it + 2, src_it + 6);

        test_name = "Regular call to std::reduce 2/2";
        failed_tests += ASSERT_EQUAL(test_name, result, 14);
    }

    // Second Group: Testing calls to std::reduce using make_transform_iterator

    {
        // test 1/3

        // create buffer
        sycl::buffer<std::tuple<bool, bool, bool>, 1> src_buf { sycl::range<1>(3) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            src[0] = std::tuple<bool, bool, bool>(false, false, false);
            src[1] = std::tuple<bool, bool, bool>(false, true, true);
            src[2] = std::tuple<bool, bool, bool>(false, false, true);
        }

        {
            // call algorithm
            auto result = std::reduce
            (
                oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::make_transform_iterator
                (
                    src_it,
                    oppositeFunctor()
                ),
                oneapi::dpl::make_transform_iterator
                (
                    src_end_it,
                    oppositeFunctor()
               ),
                std::tuple<bool, bool, bool>(true, true, true),
                andBooleanFunctor()
            );

            test_name = "std::reduce with make_transform_iterator and tuple output";
            failed_tests += ASSERT_EQUAL(test_name, std::get<0>(result), true);
            failed_tests += ASSERT_EQUAL(test_name, std::get<1>(result), false);
            failed_tests += ASSERT_EQUAL(test_name, std::get<2>(result), false);
        }


        // test 2/3

        // call algorithm
        auto result = std::reduce
        (
            oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::make_transform_iterator
            (
                dpct::make_counting_iterator(4),
                square()
            ),
            oneapi::dpl::make_transform_iterator
            (
                dpct::make_counting_iterator(8),
                square()
            )
        );

        test_name = "std::reduce with make_transform_iterator 2";
        failed_tests += ASSERT_EQUAL(test_name, result, 126);

        // test 3/3

        // create buffer
        sycl::buffer<uint64_t, 1> rand_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> field_buf { sycl::range<1>(8) };

        auto rand_it = oneapi::dpl::begin(rand_buf);
        auto field_it = oneapi::dpl::begin(field_buf);

        {
            auto rand = rand_buf.template get_access<sycl::access::mode::write>();
            auto field = field_buf.template get_access<sycl::access::mode::write>();

            rand[0] = 4; rand[1] = 1; rand[2] = 9; rand[3] = 8; rand[4] = 5; rand[5] = 3; rand[6] = 0; rand[7] = 7;

            field[0] = 7; field[1] = 3; field[2] = 2; field[3] = 4; field[4] = 0; field[5] = 9; field[6] = 8; field[7] = 5;

            // rand: { 4, 1, 9, 8, 5, 3, 0, 7 }
            // field: { 7, 3, 2, 4, 0, 9, 8, 5 }
        }


        // call algorithm
        result = std::reduce
        (
            oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::make_transform_iterator
            (
                oneapi::dpl::make_zip_iterator
                (
                    rand_it + 2,
                    field_it + 2
                ),
                multiply()
            ),
            oneapi::dpl::make_transform_iterator
            (
                oneapi::dpl::make_zip_iterator
                (
                    rand_it + 6,
                    field_it + 6
                ),
                multiply()
            ),
            0.0,
            std::plus<int>()
        );

        test_name = "std::reduce with make_transform_iterator 3";
        failed_tests += ASSERT_EQUAL(test_name, result, 77);
    } 
    // Third Group: Testing calls to dpct::reduce_argmin and dpct::reduce_argmax

    {
        // test 1/1

        // create input data
        std::vector<uint64_t> input(10);
        input[0] = 7;
        input[1] = 4;
        input[2] = 7;
        input[3] = 9;
        input[4] = 0;
        input[5] = 9;
        input[6] = 3;
        input[7] = 1;
        input[8] = 0;
        input[9] = 1;
        auto queue = dpct::get_default_queue();
        uint64_t* dev_input = sycl::malloc_device<uint64_t>(10, queue);
        dpct::key_value_pair<int, uint64_t> output(0,0);
        dpct::key_value_pair<int, uint64_t>* dev_output = sycl::malloc_device<dpct::key_value_pair<int, uint64_t>>(1, queue);

        queue.memcpy(dev_input, input.data(), 10 * sizeof(uint64_t)).wait();

        // call algorithm
        dpct::reduce_argmax(oneapi::dpl::execution::make_device_policy(queue), dev_input, dev_output, 10);

        queue.memcpy(&output, dev_output, 1 * sizeof(dpct::key_value_pair<int_fast64_t, uint64_t>)).wait();

        test_name = "dpct::reduce_argmax with uint64_t";
        failed_tests += ASSERT_EQUAL(test_name, output.key, 3) || ASSERT_EQUAL(test_name, output.value, 9);

        // call algorithm
        dpct::reduce_argmin(oneapi::dpl::execution::make_device_policy(queue), dev_input, dev_output, 10);

        queue.memcpy(&output, dev_output, 1 * sizeof(dpct::key_value_pair<int, uint64_t>)).wait();

        test_name = "dpct::reduce_argmin with uint64_t";
        failed_tests += ASSERT_EQUAL(test_name, output.key, 4) || ASSERT_EQUAL(test_name, output.value, 0);

    }

    // Third Group: Testing calls to dpct::segmented_argmin 
    {
        // test 1/1

        // create input data
        ::std::vector<::std::uint64_t> input(30);
        input[0] = 7; //0
        input[1] = 4; //1
        input[2] = 7; //1
        input[3] = 9; //1
        input[4] = 0; //1
        input[5] = 9; //1
        input[6] = 3; //1
        input[7] = 1; //2
        input[8] = 0; //2
        input[9] = 1; //2
        input[10] = 3; //3
        input[11] = 3; //4
        input[12] = 3; //4
        input[13] = 4; //4
        input[14] = 8; //4
        input[15] = 2; //4
        input[16] = 1; //4
        input[17] = 0; //5
        input[18] = 0; //5
        input[19] = 6; //5
        input[20] = 5; //5
        input[21] = 3; //5
        input[22] = 3; //6
        input[23] = 4; //6
        input[24] = 7; //6
        input[25] = 7; //7
        input[26] = 1; //7
        input[27] = 0; //7
        input[28] = 6; //7
        input[29] = 2; //7

        ::std::vector<::std::uint64_t> offsets(10);
        offsets[0] = 0;
        offsets[1] = 1;
        offsets[2] = 7;
        offsets[3] = 10;
        offsets[4] = 11;
        offsets[5] = 17;
        offsets[6] = 22;
        offsets[7] = 25;
        offsets[8] = 25;
        offsets[9] = 30;

        auto queue = dpct::get_default_queue();
        ::std::uint64_t* dev_input = sycl::malloc_device<::std::uint64_t>(30, queue);
        ::std::uint64_t* dev_offsets = sycl::malloc_device<::std::uint64_t>(10, queue);
        ::std::vector<dpct::key_value_pair<int, ::std::uint64_t>> output(9);

        dpct::key_value_pair<int, ::std::uint64_t>* dev_output = 
                sycl::malloc_device<dpct::key_value_pair<int, ::std::uint64_t>>(9, queue);

        queue.memcpy(dev_input, input.data(), 30 * sizeof(::std::uint64_t)).wait();
        queue.memcpy(dev_offsets, offsets.data(), 10 * sizeof(::std::uint64_t)).wait();
        queue.memcpy(dev_output, output.data(), 9 * sizeof(dpct::key_value_pair<int, ::std::uint64_t>)).wait();

        // call algorithm
        dpct::segmented_reduce_argmax(oneapi::dpl::execution::make_device_policy(queue), dev_input, dev_output, 
                                      9, dev_offsets, dev_offsets+1);
        
        queue.memcpy(output.data(), dev_output, 9 * sizeof(dpct::key_value_pair<int, ::std::uint64_t>)).wait();

        test_name = "dpct::segmented_reduce_argmax with ::std::uint64_t";
        failed_tests += ASSERT_EQUAL(test_name, output[0].key, 0) || ASSERT_EQUAL(test_name, output[0].value, 7);
        failed_tests += ASSERT_EQUAL(test_name, output[1].key, 2) || ASSERT_EQUAL(test_name, output[1].value, 9);
        failed_tests += ASSERT_EQUAL(test_name, output[2].key, 0) || ASSERT_EQUAL(test_name, output[2].value, 1);
        failed_tests += ASSERT_EQUAL(test_name, output[3].key, 0) || ASSERT_EQUAL(test_name, output[3].value, 3);
        failed_tests += ASSERT_EQUAL(test_name, output[4].key, 3) || ASSERT_EQUAL(test_name, output[4].value, 8);
        failed_tests += ASSERT_EQUAL(test_name, output[5].key, 2) || ASSERT_EQUAL(test_name, output[5].value, 6);
        failed_tests += ASSERT_EQUAL(test_name, output[6].key, 2) || ASSERT_EQUAL(test_name, output[6].value, 7);
        failed_tests += ASSERT_EQUAL(test_name, output[7].key, 1) || 
                        ASSERT_EQUAL(test_name, output[7].value, ::std::numeric_limits<::std::uint64_t>::lowest());
        failed_tests += ASSERT_EQUAL(test_name, output[8].key, 0) || ASSERT_EQUAL(test_name, output[8].value, 7);

        // call algorithm
        dpct::segmented_reduce_argmin(oneapi::dpl::execution::make_device_policy(queue), dev_input, dev_output, 9, 
                                      dev_offsets, dev_offsets+1);

        queue.memcpy(output.data(), dev_output, 9 * sizeof(dpct::key_value_pair<int, ::std::uint64_t>)).wait();

        test_name = "dpct::segmented_reduce_argmin with ::std::uint64_t";
        failed_tests += ASSERT_EQUAL(test_name, output[0].key, 0) || ASSERT_EQUAL(test_name, output[0].value, 7);
        failed_tests += ASSERT_EQUAL(test_name, output[1].key, 3) || ASSERT_EQUAL(test_name, output[1].value, 0);
        failed_tests += ASSERT_EQUAL(test_name, output[2].key, 1) || ASSERT_EQUAL(test_name, output[2].value, 0);
        failed_tests += ASSERT_EQUAL(test_name, output[3].key, 0) || ASSERT_EQUAL(test_name, output[3].value, 3);
        failed_tests += ASSERT_EQUAL(test_name, output[4].key, 5) || ASSERT_EQUAL(test_name, output[4].value, 1);
        failed_tests += ASSERT_EQUAL(test_name, output[5].key, 0) || ASSERT_EQUAL(test_name, output[5].value, 0);
        failed_tests += ASSERT_EQUAL(test_name, output[6].key, 0) || ASSERT_EQUAL(test_name, output[6].value, 3);
        failed_tests += ASSERT_EQUAL(test_name, output[7].key, 1) || 
                        ASSERT_EQUAL(test_name, output[7].value, ::std::numeric_limits<::std::uint64_t>::max());
        failed_tests += ASSERT_EQUAL(test_name, output[8].key, 2) || ASSERT_EQUAL(test_name, output[8].value, 0);

        sycl::free(dev_input, queue);
        sycl::free(dev_offsets, queue);
        sycl::free(dev_output, queue);
        
    }

    // Testing calls to dpct::argmin and dpct::argmax functors with unique and equivalent values
    {
        auto queue = dpct::get_default_queue();
        test_name = "oneapi::dpl::reduce with dpct::argmin functor - All values are unique";
        std::size_t n = 10;
        sycl::buffer<dpct::key_value_pair<int, float>> input(n);
        {
            auto host_acc = input.get_host_access();
            for (std::size_t i = 0; i < n; ++i)
                host_acc[i].key = i + 10, host_acc[i].value = i + 20;
            // inject min and max
            host_acc[4].key = 9;
            host_acc[4].value = 8;
            host_acc[6].key = 101;
            host_acc[6].value = 99;
        }
        auto argmin_res = oneapi::dpl::reduce(dpl::execution::make_device_policy(queue),
                                              oneapi::dpl::begin(input),
                                              oneapi::dpl::end(input),
                                              dpct::key_value_pair<int, float>(std::numeric_limits<int>::max(),
                                                                               std::numeric_limits<float>::max()),
                                              dpct::argmin());

        failed_tests += ASSERT_EQUAL(test_name, argmin_res.key, 9);
        failed_tests += ASSERT_EQUAL(test_name, argmin_res.value, 8);

        test_name = "oneapi::dpl::reduce with dpct::argmax functor - All values are unique";
        auto argmax_res = oneapi::dpl::reduce(dpl::execution::make_device_policy(queue),
                                              oneapi::dpl::begin(input),
                                              oneapi::dpl::end(input),
                                              dpct::key_value_pair<int, float>(std::numeric_limits<int>::min(),
                                                                               std::numeric_limits<float>::min()),
                                              dpct::argmax());

        failed_tests += ASSERT_EQUAL(test_name, argmax_res.key, 101);
        failed_tests += ASSERT_EQUAL(test_name, argmax_res.value, 99);

        test_name = "oneapi::dpl::reduce with dpct::argmin functor - All values are the same";
        {
            auto host_acc = input.get_host_access();
            for (std::size_t i = 0; i < n; ++i)
                host_acc[i].key = i + 30, host_acc[i].value = 2;
        }
        // Expect the key_value_pair with the lower key to be returned when value compares equal
        argmin_res = oneapi::dpl::reduce(dpl::execution::make_device_policy(queue),
                                         oneapi::dpl::begin(input),
                                         oneapi::dpl::end(input),
                                         dpct::key_value_pair<int, float>(std::numeric_limits<int>::max(),
                                                                          std::numeric_limits<float>::max()),
                                         dpct::argmin());

        failed_tests += ASSERT_EQUAL(test_name, argmin_res.key, 30);
        failed_tests += ASSERT_EQUAL(test_name, argmin_res.value, 2);

        argmax_res = oneapi::dpl::reduce(dpl::execution::make_device_policy(queue),
                                         oneapi::dpl::begin(input),
                                         oneapi::dpl::end(input),
                                         dpct::key_value_pair<int, float>(std::numeric_limits<int>::min(),
                                                                          std::numeric_limits<float>::min()),
                                         dpct::argmax());

        // Expect the key_value_pair with the lower key to be returned when value compares equal
        failed_tests += ASSERT_EQUAL(test_name, argmax_res.key, 30);
        failed_tests += ASSERT_EQUAL(test_name, argmax_res.value, 2);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
