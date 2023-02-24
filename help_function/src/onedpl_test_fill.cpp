// ====------ onedpl_test_fill.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE

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

template<typename Buffer> void init_buffer(Buffer& src, int start_index, int end_index, uint64_t value) {
    for (int i = start_index; i != end_index; ++i) {
        src[i] = value;
    }
}

template<typename Vector> void iota_vector(Vector& vec, int start_index, int end_index) {
    for (int i = start_index; i != end_index; ++i) {
        vec[i] = i;
    }
}

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    // First 3 tests: Testing regular call to std::fill
    {
        // test 1/3

        // create buffer
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            init_buffer(src, 0, 8, 0);
        }

        // call algorithm
        std::fill(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, 2);

        {
            std::string test_name = "Regular call to std::fill 1/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src[i], 2);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/3

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            init_buffer(src, 0, 8, 0);
        }

        // call algorithm:
        std::fill(oneapi::dpl::execution::dpcpp_default, src_it + 2, src_end_it, 5);

        {
            std::string test_name = "Regular call to std::fill 2/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 1)
                    num_failing += ASSERT_EQUAL(test_name, src[i], 5);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/3

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            init_buffer(src, 0, 8, 0);
        }

        // call algorithm
        std::fill(oneapi::dpl::execution::dpcpp_default, src_it + 2, src_it + 6, 3);

        {
            std::string test_name = "Regular call to std::fill 3/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 1 && i < 6)
                    num_failing += ASSERT_EQUAL(test_name, src[i], 3);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Second 3 tests: Testing call to std::fill using make_permutation_iterator

    {
        // test 1/3

        // create buffer
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf{ sycl::range<1>(4) };

        {
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                src[i] = i;
            }
            map[0] = 7; map[1] = 6; map[2] = 5; map[3] = 4;
            // src buffer: { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map buffer: { 7, 6, 5, 4 }
        }

        auto src_it = oneapi::dpl::begin(src_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_end, 20);
        }

        {
            std::string test_name = "std::fill with perm_it 1/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], 20);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/3

        {
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                src[i] = i;
            }
            map[0] = 3; map[1] = 0; map[2] = 2; map[3] = 1;
            // src buffer: { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map buffer: { 3, 0, 2, 1 }
        }


        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_end, 20);
        }

        {
            std::string test_name = "std::fill with perm_it 2/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src[i], 20);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/3

        {
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                src[i] = i;
            }
            map[0] = 2; map[1] = 4; map[2] = 3; map[3] = 5;
            // src buffer: { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map buffer: { 2, 4, 3, 5 }
        }


        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_end, 20);
        }

        {
            std::string test_name = "std::fill with perm_it 3/3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 1 && i < 6)
                    num_failing += ASSERT_EQUAL(test_name, src[i], 20);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Third 3 tests: Testing call to std::fill using device_pointer<T>
    
    // These tests assume USM is available, disable when it isn't
#ifndef DPCT_USM_LEVEL_NONE 
    {
        // test 1/3

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);

        // fill hostArray with 0s
        init_buffer(hostArray, 0, 8, 0);

        myQueue.submit([&](sycl::handler& h) {
            // copy hostArray to deviceArray
            h.memcpy(deviceArray, hostArray, 8 * sizeof(int));
        });
        myQueue.wait();

        {
            auto dptr_begin = dpct::device_pointer<int>(deviceArray);
            auto dptr_end = dpct::device_pointer<int>(deviceArray + 4);

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, dptr_begin, dptr_end, 12);
        }

        myQueue.submit([&](sycl::handler& h) {
            // copy deviceArray back to hostArray
            h.memcpy(hostArray, deviceArray, 8 * sizeof(int));
        });
        myQueue.wait();

        std::string test_name = "std::fill with device_pointer<T> 1/3";
        for (int i = 0; i != 8; ++i) {
            if (i < 4)
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 12);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 0);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;

        // test 2/3

        // fill hostArray again with 0s
        init_buffer(hostArray, 0, 8, 0);

        myQueue.submit([&](sycl::handler& h) {
            // copy hostArray to deviceArray
            h.memcpy(deviceArray, hostArray, 8 * sizeof(int));
        });
        myQueue.wait();

        {
            auto dptr_begin = dpct::device_pointer<int>(deviceArray + 4);
            auto dptr_end = dpct::device_pointer<int>(deviceArray + 8);

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, dptr_begin, dptr_end, 12);
        }

        myQueue.submit([&](sycl::handler& h) {
            // copy deviceArray back to hostArray
            h.memcpy(hostArray, deviceArray, 8 * sizeof(int));
        });
        myQueue.wait();

        test_name = "std::fill with device_pointer<T> 2/3";
        for (int i = 0; i != 8; ++i) {
            if (i > 3)
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 12);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 0);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;

        // test 3/3

        // fill hostArray again with 0s
        init_buffer(hostArray, 0, 8, 0);

        myQueue.submit([&](sycl::handler& h) {
            // copy hostArray to deviceArray
            h.memcpy(deviceArray, hostArray, 8 * sizeof(int));
        });
        myQueue.wait();

        {
            auto dptr_begin = dpct::device_pointer<int>(deviceArray + 2);
            auto dptr_end = dpct::device_pointer<int>(deviceArray + 6);

            // call algorithm
            std::fill(oneapi::dpl::execution::dpcpp_default, dptr_begin, dptr_end, 12);
        }

        myQueue.submit([&](sycl::handler& h) {
            // copy deviceArray back to hostArray
            h.memcpy(hostArray, deviceArray, 8 * sizeof(int));
        });
        myQueue.wait();

        test_name = "std::fill with device_pointer<T> 3/3";
        for (int i = 0; i != 8; ++i) {
            if (i > 1 && i < 6)
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 12);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 0);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    // std::fill_n tests

    {
        // test 1/2: call to std::fill_n using device_pointer<T>

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        float hostArray[8];
        float *deviceArray = (float*) malloc_device(8 * sizeof(float), dev, ctxt);

        // fill hostArray with 0s
        iota_vector(hostArray, 0, 8);

        myQueue.submit([&](sycl::handler& h) {
            // copy hostArray to deviceArray
            h.memcpy(deviceArray, hostArray, 8 * sizeof(float));
        });
        myQueue.wait();

        {
            auto dptr_begin = dpct::device_pointer<float>(deviceArray + 2);

            // call algorithm
            std::fill_n(oneapi::dpl::execution::make_device_policy(myQueue), dptr_begin, 4, 10.5f);
        }

        myQueue.submit([&](sycl::handler& h) {
            // copy deviceArray back to hostArray
            h.memcpy(hostArray, deviceArray, 8 * sizeof(float));
        });
        myQueue.wait();

        std::string test_name = "std::fill_n with device_pointer<T> 1/2";
        for (int i = 0; i != 8; ++i) {
            if (i > 1 && i < 6)
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 10.5);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }
#endif //DPCT_USM_LEVEL_NONE

    {
        // test 2/2: call to std::fill_n using device_vector<T>

        // create device_vector and src vector
        std::vector<int> src(8);

        iota_vector(src, 0, 8);
        dpct::device_vector<int> dv(src);

        dpct::get_default_queue().wait();

        {
            // call algorithm on dv
            std::fill_n(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dv.begin(), 4, 10);
        }

        std::string test_name = "std::fill_n with device_pointer<T> 2/2";
        for (int i = 0; i != 8; ++i) {
            if (i < 4)
                num_failing += ASSERT_EQUAL(test_name, dv[i], 10);
            else
                num_failing += ASSERT_EQUAL(test_name, dv[i], i);
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
