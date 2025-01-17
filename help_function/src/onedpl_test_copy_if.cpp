// ====------ onedpl_test_copy_if.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>
#include <iomanip>

#include <sycl/sycl.hpp>

struct is_odd
{
    bool operator()(const int64_t &x) const {
        return x % 2;
    }
};

struct is_even
{
    bool operator()(const int64_t &x) const {
        return (x % 2) == 0;
    }
};

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

class CopyIf1 {};     // name for policy
class CopyIf2 {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    // #77 COPY IF TEST //

    {
        // create buffer
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> stn_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        auto stn_it = oneapi::dpl::begin(stn_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();

            for (int i=0; i != 8; ++i) {
                src[i] = i;
                stn[i] = i % 2; // copy every other element, beginning with first.
                dst[i] = 0;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<CopyIf1>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        dpct::copy_if(new_policy, src_it, src_end_it, stn_it, dst_it, oneapi::dpl::identity());

        {
            std::string test_name = "Regular call to dpct::copy_if";
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::read>();
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i=0; i != 8; ++i) {
                num_failing += ASSERT_EQUAL(test_name, dst[i], i < 4 ? i*2 + 1 : 0);
                num_failing += ASSERT_EQUAL(test_name, stn[i], i % 2);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // create buffer
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();

            for (int i=0; i != 8; ++i) {
                src[i] = i;
                dst[i] = 0;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<CopyIf2>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::copy_if(new_policy, src_it, src_end_it, dst_it, is_odd());

        {
            std::string test_name = "Regular call to std::copy_if";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            int pos = 0;
            for (int i=0; i != 8; ++i) {
                if (i%2 == 1) {
                    num_failing += ASSERT_EQUAL(test_name, dst[pos], i);
                    ++pos;
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Adding 2 more tests below

    // dpct::copy_if with make_permutation_iterator

    {
        // create buffers
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> stn_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto map_it = oneapi::dpl::begin(map_buf);
        auto stn_it = oneapi::dpl::begin(stn_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                src[i] = i;
                dst[i] = 0;
            }

            map[0] = 1; map[1] = 3; map[2] = 2; map[3] = 6;
            stn[0] = -3; stn[1] = -2; stn[2] = 0; stn[3] = 4; stn[4] = 0; stn[5] = 2; stn[6] = 1; stn[7] = 3;

            // src: { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map: { 1, 3, 2, 6 }
            // stn: { -3, -2, 0, 4, 0, 2, 1, 3 }
            // dst: { 0, 0, 0, 0, 0, 0, 0, 0 }
        }

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm:
            dpct::copy_if
            (
                oneapi::dpl::execution::dpcpp_default,
                perm_begin,
                perm_end,
                stn_it,
                dst_it,
                is_even()
            );
        }
        // expected dst: { 3, 2, 6, 0, 0, 0, 0, 0 }

        std::string test_name = "copy_if with perm_it";
        auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

        for (int i = 0; i != 8; ++i) {
            if (i == 0)
                num_failing += ASSERT_EQUAL(test_name, dst[i], 3);
            else if (i == 1)
                num_failing += ASSERT_EQUAL(test_name, dst[i], 2);
            else if (i == 2)
                num_failing += ASSERT_EQUAL(test_name, dst[i], 6);
            else
                num_failing += ASSERT_EQUAL(test_name, dst[i], 0);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    // dcpt::copy_if with device_vector

    {

        // create src and dst device_vector
        std::vector<int> src_vec(8);
        std::vector<int> dst_vec(8);

        for (int i = 0; i != 8; ++i) {
            src_vec[i] = i;
            dst_vec[i] = 0;
        }

        dpct::device_vector<int> src_dv(src_vec);
        dpct::device_vector<int> dst_dv(dst_vec);

        // src_dv: { 0, 1, 2, 3, 4, 5, 6, 7 }
        // dst_dv: { 0, 0, 0, 0, 0, 0, 0, 0 }

        {
            // call algorithm
            dpct::copy_if
            (
                oneapi::dpl::execution::dpcpp_default,
                src_dv.begin() + 4,
                src_dv.begin() + 8,
                dpct::make_counting_iterator(4),
                dst_dv.begin(),
                ([=](int i) {
                    return (i % 2) == 1;
                })
            );
        }

        std::string test_name = "copy_if with device_vector";

        // expected dst_vec: { 5, 7, 0, 0, 0, 0, 0, 0 }
        for (int i = 0; i != 8; ++i) {
            if (i == 0)
                num_failing += ASSERT_EQUAL(test_name, dst_dv[i], 5);
            else if (i == 1)
                num_failing += ASSERT_EQUAL(test_name, dst_dv[i], 7);
            else
                num_failing += ASSERT_EQUAL(test_name, dst_dv[i], 0);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
