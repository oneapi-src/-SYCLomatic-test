// ====------ onedpl_test_transform_reduce.cpp---------- -*- C++ -* ----===////
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

struct square {
    double operator()(double& x) const {
        return x*x;
    }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    std::string test_name = "";

    // test 1/2

    // create queue
    sycl::queue myQueue;
    auto dev = myQueue.get_device();
    auto ctxt = myQueue.get_context();

    // create host and device vectors
    std::vector<double> host_vec(8);

    // host_vec = { 0, 1, 2, 3, 4, 5, 6, 7 }
    for (int i = 0; i != 8; ++i) {
        host_vec[i] = i;
    }


    {
        dpct::device_vector<double> dev_vec(host_vec);

        // call algorithm
        auto result = std::transform_reduce(oneapi::dpl::execution::dpcpp_default, dev_vec.begin(), dev_vec.begin()+4, 0., std::plus<double>(), square());

        test_name = "transform_reduce with device_vector";
        failed_tests += ASSERT_EQUAL(test_name, result, 14);
    }

    // test 2/2

    std::vector<bool> host_vec2(8);

    // host_vec2 = { 1, 1, 1, 1, 0, 0, 0, 0 }
    for (int i = 0; i != 8; ++i) {
        if (i < 4)
            host_vec2[i] = 1;
        else
            host_vec2[i] = 0;
    }

    {
        dpct::device_vector<bool> dev_vec2(host_vec2);


        // call algorithm
        auto result = std::transform_reduce(oneapi::dpl::execution::dpcpp_default, dev_vec2.begin()+4, dev_vec2.end(), 0, std::plus<bool>(), oneapi::dpl::identity());

        test_name = "transform_reduce with device_vector 2";
        failed_tests += ASSERT_EQUAL(test_name, result, 0);

    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
