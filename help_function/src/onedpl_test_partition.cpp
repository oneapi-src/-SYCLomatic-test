// ====------ onedpl_test_partition.cpp---------- -*- C++ -* ----===////
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

#include <sycl/sycl.hpp>

#include <iostream>
#include <iomanip>

struct is_odd
{
    bool operator()(const int64_t &x) const {
        return x % 2;
    }
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

class Partition1 {};     // name for policy
class Partition2 {};     // name for policy

int main() {

    // #78 PARTITION TEST //

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        // create buffer
        sycl::buffer<int64_t, 1> input_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> stencil_buf{ sycl::range<1>(8) };

        auto inp_it = oneapi::dpl::begin(input_buf);
        auto inp_end_it = oneapi::dpl::end(input_buf);
        auto stn_it = oneapi::dpl::begin(stencil_buf);

        {
            auto inp = inp_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                inp[i] = i;
                stn[i] = is_odd{}(i);
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Partition2>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::partition(new_policy, inp_it, inp_end_it, stn_it, oneapi::dpl::identity());

        {
            test_name = "Regular call to dpct::partition";
            auto inp = inp_it.get_buffer().template get_access<sycl::access::mode::write>();

            // requires no falses before any trues, but no specific order.
            bool entered_falses = false;
            
            for (int i = 0; i < 8; i ++)
            {
                if (!is_odd{}(inp[i]))
                    entered_falses = true;
                num_failing += ASSERT_EQUAL(test_name, !(entered_falses && is_odd{}(inp[i])), true);
            }
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
