// ====------ onedpl_test_sort_by_key.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>
#include <iomanip>

template<typename Iterator, typename T>
bool check_values(Iterator first, Iterator last, const T& val)
{
    return std::all_of(first, last,
        [&val](const T& x) { return x == val; });
}

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

template<typename Predicate>
struct unique_by_key_fun {
    typedef bool result_of;
    unique_by_key_fun(Predicate _pred) : pred(_pred) {}
    template<typename _T1, typename _T2> result_of operator() (_T1&& a, _T2&& b) const { return pred(std::get<0>(a),std::get<0>(b)); }
private:
    Predicate pred;
};

struct Identity {
    template <typename T>
    T operator()(const T& x) const {
        return x;
    }
};

struct Negate
{
    template <typename T>
    T operator()(const T& x) const
    {
        return -x;
    }
};

struct Inc
{
    template <typename T>
    void operator()(T& x) const
    {
        ++x;
    }
};

struct Plus
{
    template <typename T, typename U>
    T operator()(const T x, const U y) const
    {
        return x + y;
    }
};

struct EqualTo
{
    template <typename _T1, typename _T2>
    bool operator()(const _T1& x, const _T2& y) const {
        return x == y;
    }
};

class StableSortByKey {};     // name for policy
class GatherIf {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #14 SORT BY KEY TEST //

    {
        sycl::queue myQueue;
        const int N = 6;
        sycl::buffer<int, 1> keys_buf{ sycl::range<1>(N) };
        sycl::buffer<int, 1> values_buf{ sycl::range<1>(N) };

        auto keys_it = oneapi::dpl::begin(keys_buf);
        auto values_it = oneapi::dpl::begin(values_buf);

        {
            auto keys = keys_it.get_buffer().get_host_access();
            auto values = values_it.get_buffer().get_host_access();

            keys[0] = 1; keys[1] = 4; keys[2] = 2; keys[3] = 8; keys[4] = 5; keys[5] = 7;
            values[0] = 'a'; values[1] = 'b'; values[2] = 'c'; values[3] = 'd'; values[4] = 'e';values[5] = 'f';
        }

        // call algorithm:
        dpct::sort(oneapi::dpl::execution::make_device_policy<class kernel1>(myQueue), keys_it, keys_it + N, values_it);

        // keys is now   {  1,   2,   4,   5,   7,   8}
        // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
        {
            test_name = "Regular call to sort";
            auto values = values_it.get_buffer().get_host_access();
            num_failing += ASSERT_EQUAL(test_name, values[0], 'a');
            num_failing += ASSERT_EQUAL(test_name, values[1], 'c');
            num_failing += ASSERT_EQUAL(test_name, values[2], 'b');
            num_failing += ASSERT_EQUAL(test_name, values[3], 'e');
            num_failing += ASSERT_EQUAL(test_name, values[4], 'f');
            num_failing += ASSERT_EQUAL(test_name, values[5], 'd');

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // These tests assume USM is available, disable when it isn't
#ifndef DPCT_USM_LEVEL_NONE
    {
    // Test One, call to dpct::sort using USM allocations
        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create keys array and values array
        int *keysArray = (int*) malloc_device(10 * sizeof(int), dev, ctxt);
        int *valuesArray = (int*) malloc_device(10 * sizeof(int), dev, ctxt);

        // load in data
        int keys_data[10] = {5, 8, 9, 1, 0, 6, 2, 7, 3, 4};
        int values_data[10] = {'c', 'i', 'f', 'e', 'h', 'g', 'b', 'd', 'j', 'a'};
        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(keysArray, keys_data, 10 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(valuesArray, values_data, 10 * sizeof(int));
        });
        myQueue.wait();
        {
            auto keys_begin = dpct::device_pointer<int>(keysArray);
            auto keys_end = dpct::device_pointer<int>(keysArray + 10);
            auto values_begin = dpct::device_pointer<int>(valuesArray);
            // call algorithm
            dpct::sort(oneapi::dpl::execution::make_device_policy<class kernel2>(myQueue), keys_begin, keys_end, values_begin);
        }

        // copy back
        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(keys_data, keysArray, 10 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(values_data, valuesArray, 10 * sizeof(int));
        });
        myQueue.wait();
        // check result
        int check_values[10] = {'h', 'e', 'b', 'j', 'a', 'c', 'g', 'd', 'i', 'f'};
        int check_keys[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        test_name = "sort using USM allocations";

        for (int i = 0; i != 10; ++i) {
            num_failing += ASSERT_EQUAL(test_name, values_data[i], check_values[i]);
        }

        for (int i = 0; i != 10; ++i) {
            num_failing += ASSERT_EQUAL(test_name, keys_data[i], check_keys[i]);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
#endif //DPCT_USM_LEVEL_NONE

    {
    // Test Two, test calls to dpct::sort using device vectors
        std::vector<int> keys_data{4, 8, 5, 3, 0, 9, 7, 2, 1, 6};
        std::vector<int> values_data{13, 16, 17, 11, 19, 14, 12, 18, 10, 15};

        dpct::device_vector<int> keys_vec(keys_data);
        dpct::device_vector<int> values_vec(values_data);

        auto keys_it = keys_vec.begin();
        auto keys_it_end = keys_vec.end();
        auto values_it = values_vec.begin();
        {
            // call algorithm
            dpct::sort(oneapi::dpl::execution::dpcpp_default, keys_it, keys_it_end, values_it);
            // keys is now = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            // values is now = {19, 10, 18, 11, 13, 17, 15, 12, 16, 14}
        }


            // check that values and keys are correct

            for (int i = 0; i != 10; ++i) {
                num_failing += ASSERT_EQUAL(test_name, values_vec[i], check_values[i]);
                num_failing += ASSERT_EQUAL(test_name, keys_vec[i], check_keys[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    // #15 STABLE SORT BY KEY TEST //

    {
        const int N = 6;
        sycl::buffer<int, 1> keys_buf{ sycl::range<1>(N) };
        sycl::buffer<int, 1> values_buf{ sycl::range<1>(N) };

        auto keys_it = oneapi::dpl::begin(keys_buf);
        auto values_it = oneapi::dpl::begin(values_buf);

        {
            auto keys = keys_it.get_buffer().get_host_access();
            auto values = values_it.get_buffer().get_host_access();
            keys[0] = 1; keys[1] = 4; keys[2] = 2; keys[3] = 8; keys[4] = 5; keys[5] = 7;
            values[0] = 'a'; values[1] = 'b'; values[2] = 'c'; values[3] = 'd'; values[4] = 'e';values[5] = 'f';
        }

        // call algorithm:
        dpct::stable_sort(oneapi::dpl::execution::dpcpp_default, keys_it, keys_it + N, values_it);

        // keys is now   {  1,   2,   4,   5,   7,   8}
        // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
        {
            test_name = "Regular call to stable_sort";
            auto values = values_it.get_buffer().get_host_access();

            num_failing += ASSERT_EQUAL(test_name, values[0], 'a');
            num_failing += ASSERT_EQUAL(test_name, values[1], 'c');
            num_failing += ASSERT_EQUAL(test_name, values[2], 'b');
            num_failing += ASSERT_EQUAL(test_name, values[3], 'e');
            num_failing += ASSERT_EQUAL(test_name, values[4], 'f');
            num_failing += ASSERT_EQUAL(test_name, values[5], 'd');

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // These tests assume USM is available, disable when it isn't
#ifndef DPCT_USM_LEVEL_NONE
    {
    // Test Three, call to dpct::stable_sort using USM allocations
        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create keys array and values array
        int *keysArray = (int*) malloc_device(16 * sizeof(int), dev, ctxt);
        int *valuesArray = (int*) malloc_device(16 * sizeof(int), dev, ctxt);

        // load in data
        int keys_data[16] = {0, 6, 11, 2, 7, 3, 5, 9, 8, 4, 12, 10, 14, 1, 13, 15};
        int values_data[16] = {'g', 'j', 'b', 'o', 'm', 'i', 'f', 'n', 'a', 'p', 'l', 'k', 'h', 'e', 'c', 'd'};
        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(keysArray, keys_data, 16 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(valuesArray, values_data, 16 * sizeof(int));
        });
        myQueue.wait();

        {
            auto keys_begin = dpct::device_pointer<int>(keysArray);
            auto keys_end = dpct::device_pointer<int>(keysArray + 16);
            auto values_begin = dpct::device_pointer<int>(valuesArray);
            // call algorithm
            dpct::stable_sort(oneapi::dpl::execution::make_device_policy<>(myQueue), keys_begin, keys_end, values_begin);
        }

        // copy back
        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(keys_data, keysArray, 16 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(values_data, valuesArray, 16 * sizeof(int));
        });
        myQueue.wait();

        // check result
        int check_values[16] = {'g', 'e', 'o', 'i', 'p', 'f', 'j', 'm', 'a', 'n', 'k', 'b', 'l', 'c', 'h', 'd'};
        int check_keys[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        test_name = "stable_sort with USM allocations";

        for (int i = 0; i != 16; ++i) {
            num_failing += ASSERT_EQUAL(test_name, values_data[i], check_values[i]);
        }

        for (int i = 0; i != 16; ++i) {
            num_failing += ASSERT_EQUAL(test_name, keys_data[i], check_keys[i]);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
#endif // DPCT_USM_LEVEL_NONE

    { // Test Four, test calls to dpct::stable_sort with duplicate key values
        sycl::buffer<int, 1> keys_buf{ sycl::range<1>(16) };
        sycl::buffer<int, 1> values_buf{ sycl::range<1>(16) };

        auto keys_it = oneapi::dpl::begin(keys_buf);
        auto values_it = oneapi::dpl::begin(values_buf);

        {
            auto keys = keys_it.get_buffer().get_host_access();
            auto values = values_it.get_buffer().get_host_access();
            // keys = {8, 3, 0, 2, 6, 5, 1, 8, 9, 10, 7, 4, 5, 2, 2, 10}
            keys[0] = 8; keys[1] = 3; keys[2] = 0; keys[3] = 2; keys[4] = 6; keys[5] = 5;
            keys[6] = 1; keys[7] = 8; keys[8] = 9; keys[9] = 10; keys[10] = 7; keys[11] = 4;
            keys[12] = 5; keys[13] = 2; keys[14] = 2; keys[15] = 10;
            // values = {'b', 'm', 'k', 'g', 'd', 'c', 'n', 'f', 'i', 'e', 'h', 'p', 'o', 'j', 'l', 'a'}
            values[0] = 'b'; values[1] = 'm'; values[2] = 'k'; values[3] = 'g'; values[4] = 'd'; values[5] = 'c';
            values[6] = 'n'; values[7] = 'f'; values[8] = 'i'; values[9] = 'e'; values[10] = 'h'; values[11] = 'p';
            values[12] = 'o'; values[13] = 'j'; values[14] = 'l'; values[15] = 'a';
        }

        // call algorithm:
        dpct::stable_sort(oneapi::dpl::execution::dpcpp_default, keys_it, keys_it + 16, values_it);

        // keys is now = {0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10}
        // values is now = {'k', 'n', 'g', 'j', 'l', 'm', 'p', 'c', 'o', 'd', 'h', 'b', 'f', 'i', 'e', 'a'}
        {
            auto keys = keys_it.get_buffer().get_host_access();
            auto values = values_it.get_buffer().get_host_access();
            int check_values[16] = {'k', 'n', 'g', 'j', 'l', 'm', 'p', 'c', 'o', 'd', 'h', 'b', 'f', 'i', 'e', 'a'};
            int check_keys[16] = {0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10};
            // check that values and keys are correct
            test_name = "stable_sort with duplicate key values";

            for (int i = 0; i != 16; ++i) {
                num_failing += ASSERT_EQUAL(test_name, values[i], check_values[i]);
                num_failing += ASSERT_EQUAL(test_name, keys[i], check_keys[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
