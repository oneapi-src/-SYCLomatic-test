// ====------ onedpl_test_for_each.cpp---------- -*- C++ -* ----===////
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

struct square {
    void operator()(uint64_t& x) const {
        x = x*x;
    }

    // function operator used in for_each test that uses zip_iterator
    template<typename T>
    void operator()(T&& t) const {
        std::get<1>(t) = std::get<0>(t) * std::get<0>(t);
    }
};

template<class To, class From, bool compress>
struct compressFunctor
{
    To to;
    const From from;
    const int  nCmpts;

    compressFunctor
    (
        To _to,
        const From _from,
        int _nCmpts
    ):
        to(_to),
        from(_from),
        nCmpts(_nCmpts)
    {}

    void operator()(const int& i) const
    {
        if(compress)
            from[i] = (to[i] - i%nCmpts);
        else
            from[i] = (to[i] + i%nCmpts);
    }
};

template <typename T>
struct absolute_value {
    void operator()(T &x) const { x = (x >= 0 ? x : -x); }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // First Group: Testing regular call to std::for_each
    {
        // test 1/1

        // create buffer
        sycl::buffer<uint64_t> src_buf { sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);

        iota_buffer(src_buf, 0, 8);

        // call algorithm
        std::for_each(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, square());

        {
            test_name = "Regular call to std::for_each";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src[i], i*i);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Second Group: Testing calls to std::for_each using make_counting_iterator

    {
        // test 1/2

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create arrays`
        int *srcArray = (int*) malloc_shared(8 * sizeof(int), dev, ctxt);
        int *dstArray = (int*) malloc_shared(8 * sizeof(int), dev, ctxt);

        // fill srcArray and dstArray
        for (int i = 0; i != 8; ++i) {
            srcArray[i] = i;
            dstArray[i] = 0;
        }

        // call algorithm
        std::for_each
        (
            oneapi::dpl::execution::make_device_policy<class ForEach>(myQueue),
            dpct::make_counting_iterator(0),
            dpct::make_counting_iterator(4),
            compressFunctor<int*, int*, true>
            (
                srcArray,
                dstArray,
                2
            )
        );
        myQueue.wait();

        {
            test_name = "std::for_each using make_counting_it 1/2";
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dstArray[i], srcArray[i] - i%2);
                else
                    num_failing += ASSERT_EQUAL(test_name, dstArray[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/2

        // fill srcArray and dstArray
        for (int i = 0; i != 8; ++i) {
            srcArray[i] = i;
            dstArray[i] = 0;
        }

        // call algorithm
        std::for_each
        (
            oneapi::dpl::execution::make_device_policy<class ForEach2>(myQueue),
            dpct::make_counting_iterator(2),
            dpct::make_counting_iterator(6),
            ([=](int i) {
                dstArray[i] = srcArray[i] - 5;
            })
        );
        myQueue.wait();

        {
            test_name = "std::for_each using make_counting_it 2/2";
            for (int i = 0; i != 8; ++i) {
                if (i > 1 && i < 6)
                    num_failing += ASSERT_EQUAL(test_name, dstArray[i], srcArray[i] - 5);
                else
                    num_failing += ASSERT_EQUAL(test_name, dstArray[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Third Group: Testing calls to std::for_each using make_permutation_iterator
    {
        // test 1/1

        // create buffer

        sycl::buffer<uint64_t> src_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t> map_buf { sycl::range<1>(4) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        iota_buffer(src_buf, 0, 8);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 3; map[2] = 6; map[3] = 5;
        }

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::for_each
            (
                oneapi::dpl::execution::dpcpp_default,
                perm_begin,
                perm_end,
                square()
            );
        }

        {
            test_name = "std::for_each using make_perm_it";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 3 || i == 7)
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], i*i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Fourth Group: Testing calls to std::for_each using make_zip_iterator
    {
        // test 1/2

        // create buffer
        sycl::buffer<uint64_t> src_buf { sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        iota_buffer(src_buf, 0, 8);

        // call algorithm
        std::for_each
        (
            oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::make_zip_iterator
            (
                dpct::make_counting_iterator(4),
                src_it + 4
            ),
            oneapi::dpl::make_zip_iterator
            (
                dpct::make_counting_iterator(8),
                src_it + 8
            ),
            square()
        );

        {
            test_name = "std::for_each using make_zip_it 1/2";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, src[i], i*i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/2

        // create buffer
        sycl::buffer<uint64_t> map_buf { sycl::range<1>(8) };

        auto map_it = oneapi::dpl::begin(map_buf);
        auto map_end_it = oneapi::dpl::end(map_buf);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 5; map[1] = 2; map[2] = 4; map[3] = 3;
        }

        {
            // call algorithm
            std::for_each
            (
                oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::make_zip_iterator
                (
                    dpct::make_counting_iterator(0),
                    oneapi::dpl::make_permutation_iterator
                    (
                        src_it,
                        map_it
                    )
                ),
                oneapi::dpl::make_zip_iterator
                (
                    dpct::make_counting_iterator(4),
                    oneapi::dpl::make_permutation_iterator
                    (
                        src_it,
                        map_end_it
                    )
                ),
                square()
            );
        }

        {
            test_name = "std::for_each using make_zip_it 2/2";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();

            // check that src is now { 0, 1, 1, 9, 4, 0, 36, 49 }
            num_failing += ASSERT_EQUAL(test_name, src[0], 0);
            num_failing += ASSERT_EQUAL(test_name, src[1], 1);
            num_failing += ASSERT_EQUAL(test_name, src[2], 1);
            num_failing += ASSERT_EQUAL(test_name, src[3], 9);
            num_failing += ASSERT_EQUAL(test_name, src[4], 4);
            num_failing += ASSERT_EQUAL(test_name, src[5], 0);
            num_failing += ASSERT_EQUAL(test_name, src[6], 36);
            num_failing += ASSERT_EQUAL(test_name, src[7], 49);

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Fifth Group: Testing calls to std::for_each using device_vector
    {
        // test 1/1

        // create device_vector and src vector
        std::vector<int> src(8);

        src[0] = -3; src[1] = -2; src[2] = 1; src[3] = 0; src[4] = -1; src[5] = -4; src[6] = 2; src[7] = 3;
        // src: { -3, -2, 1, 0, -1, -4, 2, 3 }
        dpct::device_vector<int> dv(src);

        {
            // call algorithm on dv
            std::for_each
            (
                oneapi::dpl::execution::dpcpp_default,
                dv.begin(),
                dv.begin() + 4,
                absolute_value<int>()
            );
        }

        // check that src is now { 3, 2, 1, 0, -1, -4, 2, 3 }
        // actual result is no change in src elements
        test_name = "std::for_each using device_vector";

        num_failing += ASSERT_EQUAL(test_name, dv[0], 3);
        num_failing += ASSERT_EQUAL(test_name, dv[1], 2);
        num_failing += ASSERT_EQUAL(test_name, dv[2], 1);
        num_failing += ASSERT_EQUAL(test_name, dv[3], 0);
        num_failing += ASSERT_EQUAL(test_name, dv[4], -1);
        num_failing += ASSERT_EQUAL(test_name, dv[5], -4);
        num_failing += ASSERT_EQUAL(test_name, dv[6], 2);
        num_failing += ASSERT_EQUAL(test_name, dv[7], 3);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
