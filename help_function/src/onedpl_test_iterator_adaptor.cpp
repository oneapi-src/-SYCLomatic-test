// ====------ onedpl_test_iterator_adaptor.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define ITERATOR_ADAPTOR_REQUIRED

#include <oneapi/dpl/execution>
#include "dpct.hpp"
#include "dpl_utils.hpp"
#include <iostream>

template <typename _T>
using Vector = dpct::device_vector<_T>;

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

template<typename String, typename _T1, typename _T2>
int ASSERT_ARRAY_EQUAL(String msg, _T1&& X, _T2&& Y) {
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


//Example usage of iterator adaptor which adapts a counting iterator to
// indirectly access some device accessible USM data, and check if there is a
// difference between consecutive elements.
class IndirectAccessDiffCheckIterator
    : public dpct::iterator_adaptor<
        IndirectAccessDiffCheckIterator, oneapi::dpl::counting_iterator<int>,
        unsigned int, dpct::use_default, dpct::use_default, unsigned int,
        dpct::use_default> {
  public:
    friend dpct::iterator_core_access;
    typedef dpct::iterator_adaptor<IndirectAccessDiffCheckIterator,
                                   oneapi::dpl::counting_iterator<int>,
                                   unsigned int,
                                   dpct::use_default,
                                   dpct::use_default,
                                   unsigned int, dpct::use_default> super_t;

    IndirectAccessDiffCheckIterator(uint64_t *hashValues)
       : super_t(dpct::make_counting_iterator(0)), hashValues(hashValues) {}

  private:
    uint64_t *hashValues;

    typename super_t::reference dereference() const 
    {
        int index = *this->base_reference();
        return (index == 0 || hashValues[index] != hashValues[index - 1]);
    }
};



int
main ()
{
    int failed_tests = 0;
    const int n = 20;
    std::vector<uint64_t> keys(n);
    std::vector<uint64_t> hash_vector(n);

    std::vector<uint64_t> count_result(n);
    std::vector<uint64_t> expected_count_result = {3, 1, 0, 1, 0};
    std::vector<unsigned int> scan_result(n);
    std::vector<unsigned int> expected_scan_result = {1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};

    for (int k = 0; k < n; k ++)
    {
        keys[k] = k / 4;
        count_result[k] = 0;
        scan_result[k] = 0;
    }

    int i = 0; 
    while (i < n)
    {
        int j;
        for (j = i; (j < i + i + 1) && (j < n); j ++)
        {
            hash_vector[j] = i;
        }
        i = j;
    }

#ifdef _VERBOSE
    std::cout<<"Input Keys:"<<std::endl;
    for (int a = 0; a < n; a++)
    {
        std::cout<<keys[a]<< " ";
    }
    std::cout<<std::endl;

    std::cout<<"Input Hash Vector:"<<std::endl;
    for (int a = 0; a < n; a++)
    {
        std::cout<<hash_vector[a]<< " ";
    }
    std::cout<<std::endl;

    IndirectAccessDiffCheckIterator flag_iter(hash_vector.data());

    std::cout<<"Flag iter on host:"<<std::endl;
    for (int a = 0; a < n; a++)
    {
        std::cout<<flag_iter[a]<< " ";
    }
    std::cout<<std::endl;
#endif //_VERBOSE

    //copy from host to device
    uint64_t* dev_keys = nullptr;
    uint64_t* dev_hash_vector = nullptr;
    uint64_t* dev_key_result = nullptr;
    uint64_t* dev_count_result = nullptr;

    //using unsigned int type instead of uint64_t because 
    // reference type of iterator is unsigned int, and oneDPL inclusive_scan 
    // wants the input / output types to match
    unsigned int* dev_scan_result = nullptr;

    sycl::queue queue = dpct::get_default_queue();

#ifdef _VERBOSE
    ::std::cout << "Target device: " <<
        queue.get_info<::sycl::info::queue::device>().get_info<
        ::sycl::info::device::name>() << ::std::endl;
#endif //_VERBOSE

    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    dev_keys = sycl::malloc_device<uint64_t>(n, queue);
    dev_hash_vector = sycl::malloc_device<uint64_t>(n, queue);
    dev_key_result = sycl::malloc_device<uint64_t>(n, queue);
    dev_count_result = sycl::malloc_device<uint64_t>(n, queue);
    dev_scan_result = sycl::malloc_device<unsigned int>(n, queue);

    queue.memcpy(dev_keys, keys.data(), n * sizeof(uint64_t)).wait();
    queue.memcpy(dev_hash_vector, hash_vector.data(), n * sizeof(uint64_t)).wait();

    IndirectAccessDiffCheckIterator dev_flag_iter(dev_hash_vector);

    auto last_ele = oneapi::dpl::reduce_by_segment(policy,
                          dev_keys, dev_keys + n,
                          dev_flag_iter, dev_key_result,
                          dev_count_result);
    int count = last_ele.second - dev_count_result;

    oneapi::dpl::inclusive_scan(policy,
                           dev_flag_iter,
                           dev_flag_iter + n,
                           dev_scan_result);

    //copy back from device to host
    queue.memcpy(count_result.data(), dev_count_result, n * sizeof(uint64_t)).wait();
    queue.memcpy(scan_result.data(), dev_scan_result,  n * sizeof(unsigned int)).wait();

    int reduce_by_seg_fails = 0;
#ifdef _VERBOSE
    std::cout<<"Count Result:"<<std::endl;
#endif //_VERBOSE
    for (int a = 0; a < count; a++)
    {
        reduce_by_seg_fails += ASSERT_ARRAY_EQUAL("reduce_by_segment result check", count_result[a], expected_count_result[a]);
#ifdef _VERBOSE
        std::cout<<count_result[a]<< " ";
#endif //_VERBOSE
    }

    failed_tests += ASSERT_EQUAL("reduce_by_segment result check", reduce_by_seg_fails, 0);

    int inclusive_scan_fails = 0;
#ifdef _VERBOSE
    std::cout<<std::endl;

    std::cout<<"Scan Result:"<<std::endl;
#endif //_VERBOSE
    for (int a = 0; a < n; a++)
    {
        failed_tests += ASSERT_ARRAY_EQUAL("inclusive_scan result check", scan_result[a], expected_scan_result[a]);
#ifdef _VERBOSE
        std::cout<<scan_result[a]<< " ";
#endif //_VERBOSE
    }
#ifdef _VERBOSE
    std::cout<<std::endl;
#endif //_VERBOSE

    failed_tests += ASSERT_EQUAL("inclusive_scan result check", inclusive_scan_fails, 0);


    sycl::free(dev_keys, queue);
    sycl::free(dev_hash_vector, queue);
    sycl::free(dev_key_result, queue);
    sycl::free(dev_count_result, queue);
    sycl::free(dev_scan_result, queue);
    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
