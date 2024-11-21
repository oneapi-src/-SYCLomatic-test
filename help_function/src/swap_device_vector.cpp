// ====------ swap_device_vector.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

template <typename Vector>
bool
verify(Vector& D, int N, int V)
{
    if (D.size() != N)
    {
        std::cout << "size mismatch" << std::endl;
        return false;
    }
    for (int i = 0; i < N; ++i)
        if (D[i] != V)
        {
            std::cout << "value mismatch " << D[i] << " != " << V << std::endl;
            return false;
        }
    return true;
}

//adding these global variables to track construction and destruction
// using custom allocators with different settings.
static int num_constructed_prop = 0;
static int num_destroyed_prop = 0;
static int num_constructed_no_prop = 0;
static int num_destroyed_no_prop = 0;

template <typename T>
class AllocWithNoSwapPropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared>
{
  public:
    AllocWithNoSwapPropagation(const sycl::context& Ctxt, const sycl::device& Dev,
                               const sycl::property_list& PropList = {})
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
    {
    }
    AllocWithNoSwapPropagation(const sycl::queue& Q, const sycl::property_list& PropList = {})
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
    {
    }

    AllocWithNoSwapPropagation(const AllocWithNoSwapPropagation& other)
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
    {
    }

    typedef ::std::false_type propagate_on_container_swap;
    static void
    construct(T* p)
    {
        ::new ((void*)p) T();
        num_constructed_no_prop++;
    }
    template <typename _Arg>
    static void
    construct(T* p, _Arg arg)
    {
        ::new ((void*)p) T(arg);
        num_constructed_no_prop++;
    }
    static void
    destroy(T* p)
    {
        p->~T();
        num_destroyed_no_prop++;
    }
};

template <typename T>
class AllocWithSwapPropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared>
{
  public:
    AllocWithSwapPropagation(const sycl::context& Ctxt, const sycl::device& Dev,
                             const sycl::property_list& PropList = {})
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
    {
    }
    AllocWithSwapPropagation(const sycl::queue& Q, const sycl::property_list& PropList = {})
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
    {
    }

    AllocWithSwapPropagation(const AllocWithSwapPropagation& other)
        : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
    {
    }

    typedef ::std::true_type propagate_on_container_swap;
    static void
    construct(T* p)
    {
        ::new ((void*)p) T();
        num_constructed_prop++;
    }
    template <typename _Arg>
    static void
    construct(T* p, _Arg arg)
    {
        ::new ((void*)p) T(arg);
        num_constructed_prop++;
    }
    static void
    destroy(T* p)
    {
        p->~T();
        num_destroyed_prop++;
    }
};

int
main(void)
{
    constexpr int N1 = 3;
    constexpr int V1 = 99;
    constexpr int N2 = 7;
    constexpr int V2 = 98;
    dpct::device_vector<int> D1(N1, V1);

  // check appropriate effect of Allocator::propagate_on_container_swap

    AllocWithNoSwapPropagation<int> alloc_no_swap_prop1(dpct::get_default_queue());
    AllocWithNoSwapPropagation<int> alloc_no_swap_prop2(dpct::get_default_queue());

    //Should allocate and construct N1 int{}
    dpct::device_vector<int, AllocWithNoSwapPropagation<int>> D2(std::move(D1), alloc_no_swap_prop1);
    if (!verify(D2, N1, V1))
    {
        std::cout << "Failed move of default allocator to AllocWithNoSwapProp" << std::endl;
        return 1;
    }

    //Should allocate and construct N2 int{}
    dpct::device_vector<int, AllocWithNoSwapPropagation<int>> D3(N2, V2, alloc_no_swap_prop2);
    //since there is no swap propagation, we only destroy the excess
    D3.swap(D2);
    if (!verify(D3, N1, V1) || !verify(D2, N2, V2))
    {
        std::cout << "Failed swap of AllocWithNoSwapProp" << std::endl;
        return 1;
    }

    if (num_constructed_no_prop != 14 && num_destroyed_no_prop != 4)
    {
        std::cout << "Allocator without swap propagation is swaping incorrectly: [14, 4] != [" 
                  << num_constructed_no_prop << ", "<< num_destroyed_no_prop << "]" << std::endl;
        return 1;
    }


    dpct::device_vector<int> D4(N1, V1);

    AllocWithSwapPropagation<int> alloc_swap_prop1(dpct::get_default_queue());
    AllocWithSwapPropagation<int> alloc_swap_prop2(dpct::get_default_queue());

    //Should allocate and construct N1 int{}
    dpct::device_vector<int, AllocWithSwapPropagation<int>> D5(std::move(D4), alloc_swap_prop1);
    if (!verify(D5, N1, V1))
    {
        std::cout << "Failed move of default allocator to AllocWithSwapProp" << std::endl;
        return 1;
    }

    //Should allocate and construct N2 int{}
    dpct::device_vector<int, AllocWithSwapPropagation<int>> D6(N2, V2, alloc_swap_prop2);
    //since there is no swap propagation, we only destroy the excess
    D6.swap(D5);
    if (!verify(D6, N1, V1) || !verify(D5, N2, V2))
    {
        std::cout << "Failed swap of AllocWithSwapProp" << std::endl;
        return 1;
    }

    if (num_constructed_prop != 10 && num_destroyed_prop != 0)
    {
        std::cout << "Allocator with swap propagation is swaping incorrectly: [10, 0] != ["
                  << num_constructed_prop << ", "<< num_destroyed_prop << "]" << std::endl;
        return 1;
    }
    return 0;
}