// ====------ onedpl_test_exclusive_scan.cpp---------- -*- C++ -* ----===////
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

int main(){
    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;


    // These tests assume USM is available, disable when it isn't
#ifndef DPCT_USM_LEVEL_NONE
    {
        // Test One, test normal call to std::exclusive_scan with std::plus<> and USM allocations
        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create source and dest arrays
        int *srcArray = (int*) malloc_device(16 * sizeof(int), dev, ctxt);
        int *destArray = (int*) malloc_device(16 * sizeof(int), dev, ctxt);

        // load in data
        int src_data[16] = {4, 0, 1, 7, 5, 3, 1, 1, 0, 3, 5, 0, 9, 3, 2, 8};
        int dest_data[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(srcArray, src_data, 16 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(destArray, dest_data, 16 * sizeof(int));
        });
        myQueue.wait();

        dpct::device_pointer<int> src_begin(srcArray);
        dpct::device_pointer<int> src_end(srcArray + 16);
        dpct::device_pointer<int> dest_begin(destArray);

        // call algorithm
        int64_t zero = 0;
        std::exclusive_scan(oneapi::dpl::execution::make_device_policy<>(myQueue), src_begin, src_end, dest_begin, zero, std::plus<int64_t>());


        // copy back
        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(src_data, srcArray, 16 * sizeof(int));
        });

        myQueue.submit([&](sycl::handler& h) {
            h.memcpy(dest_data, destArray, 16 * sizeof(int));
        });
        myQueue.wait();

        // check result
        int check_src[16] = {4, 0, 1, 7, 5, 3, 1, 1, 0, 3, 5, 0, 9, 3, 2, 8};
        int check_dest[16] = {0, 4, 4, 5, 12, 17, 20, 21, 22, 22, 25, 30, 30, 39, 42, 44};

        std::string test_name = "Normal call to std::exclusive_scan with std::plus<> and USM allocations";

        for (int i = 0; i != 16; ++i) {
            num_failing += ASSERT_EQUAL(test_name, src_data[i], check_src[i]);
        }

        for (int i = 0; i != 16; ++i) {
            num_failing += ASSERT_EQUAL(test_name, dest_data[i], check_dest[i]);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
#endif //DPCT_USM_LEVEL_NONE

    {
        // Test Two, test normal call to std::exclusive_scan with std::plus<> with overlapping source and destination
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(16) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();

            // src = {5, 7, 2, 2, 4, 2, 2, 7, 3, 8, 9, 4, 2, 0, 1, 9}
            src[0] = 5; src[1] = 7; src[2] = 2; src[3] = 2; src[4] = 4; src[5] = 2; src[6] = 2;
            src[7] = 7; src[8] = 3; src[9] = 8; src[10] = 9; src[11] = 4; src[12] = 2;
            src[13] = 0; src[14] = 1; src[15] = 9;
        }

        // call algorithm
        int64_t one = 1;
        std::exclusive_scan(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it, src_it, one, std::multiplies<int64_t>());
        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            int check_src[16] = {1, 5, 35, 70, 140, 560, 1120, 2240, 15680, 47040, 376320, 3386880, 13547520, 27095040, 0, 0};

            std::string test_name = "Normal call to std::exclusive_scan with std::plus<> and overlapping source and destination";

            for (int i = 0; i != 16; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src[i], check_src[i]);
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
