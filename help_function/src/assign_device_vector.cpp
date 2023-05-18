#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

template <typename Vector>
bool verify(Vector &D, int N, int V) {
  if (D.size() != N)
  {
    std::cout<<"size mismatch"<<std::endl;
    return false;
  }
  for (int i = 0; i < N; ++i)
    if (D[i] != V)
    {
      std::cout<<"value mismatch "<<D[i]<< " != "<<V<<std::endl;
      return false;
    }
  return true;
}


static int num_constructed_prop = 0;
static int num_destroyed_prop = 0;
static int num_constructed_no_prop = 0;
static int num_destroyed_no_prop = 0;

template <typename T>
class AllocWithNoCopyPropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
  public:
  AllocWithNoCopyPropagation(const sycl::context &Ctxt, const sycl::device &Dev,
                                     const sycl::property_list &PropList = {}) 
                       : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
  {}
  AllocWithNoCopyPropagation(const sycl::queue &Q, const sycl::property_list &PropList = {}) 
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
  {}
  
  AllocWithNoCopyPropagation(const AllocWithNoCopyPropagation& other)
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
  {}

  typedef ::std::false_type propagate_on_container_copy_assignment;
  static void construct(T *p) { ::new((void*)p) T();  num_constructed_no_prop++;}
  template <typename _Arg>
  static void construct(T *p, _Arg arg) { ::new((void*)p) T(arg); num_constructed_no_prop++;}
  static void destroy(T *p) { p->~T(); num_destroyed_no_prop++;}
  
};

template <typename T>
class AllocWithCopyPropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
  public:
  AllocWithCopyPropagation(const sycl::context &Ctxt, const sycl::device &Dev,
                                     const sycl::property_list &PropList = {})
                       : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
  {}
  AllocWithCopyPropagation(const sycl::queue &Q, const sycl::property_list &PropList = {}) 
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
  {}
  
  AllocWithCopyPropagation(const AllocWithCopyPropagation& other)
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
  {}

  typedef ::std::true_type propagate_on_container_copy_assignment;
  static void construct(T *p) { ::new((void*)p) T();  num_constructed_prop++;}
  template <typename _Arg>
  static void construct(T *p, _Arg arg) { ::new((void*)p) T(arg); num_constructed_prop++;}
  static void destroy(T *p) { p->~T(); num_destroyed_prop++;}
};

int main(void)
{
    // H has storage for 4 integers
    std::vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;

    // Copy host_vector H to device_vector D
    dpct::device_vector<int> D = H;

    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    H = D;
    if (H[0] != 99 || H[1] != 88)
      return 1;

    constexpr int N = 5;
    constexpr int V = 99;
    dpct::device_vector<int> D1(N, V);

    AllocWithNoCopyPropagation<int> alloc_no_copy_prop1(dpct::get_default_queue());
    AllocWithNoCopyPropagation<int> alloc_no_copy_prop2(dpct::get_default_queue());
    
    //should construct 5 ints from D1 in the new allocation space
    dpct::device_vector<int, AllocWithNoCopyPropagation<int>> D2(std::move(D1), alloc_no_copy_prop1);
    if (!verify(D2, N, V)) {
        std::cout<<"Failed move of default allocator to AllocWithNoCopyProp"<<std::endl;
        return 1;
    }
    //Should allocate and construct 10 int{}
    dpct::device_vector<int, AllocWithNoCopyPropagation<int>> D3(10, alloc_no_copy_prop2);
    //since there is no copy propagation, we only destroy the excess, not all 10 elements, and we copy the N elements
    D3 = D2; 
    if (!verify(D3, N, V)) {
        std::cout<<"Failed move assign of AllocWithNoCopyProp"<<std::endl;
        return 1;
    }

    if (num_constructed_no_prop != 15 && num_destroyed_no_prop != 5)
    {
        std::cout<<"Allocator without copy propagation is copying incorrectly: ["<<num_constructed_no_prop<<", "<<num_destroyed_no_prop<<"]"<<std::endl;
        return 1;
    }

    AllocWithCopyPropagation<int> alloc_copy_prop1(dpct::get_default_queue());
    AllocWithCopyPropagation<int> alloc_copy_prop2(dpct::get_default_queue());
    
    //Should allocate 5 ints from D3 in the new allocation space
    dpct::device_vector<int, AllocWithCopyPropagation<int>> D4(std::move(D3), alloc_copy_prop1);
    if (!verify(D4, N, V)) {
        std::cout<<"Failed move of AllocWithNoCopyProp to AllocWithCopyProp"<<std::endl;
        return 1;
    }
    //Should allocate and construct 10 int{}
    dpct::device_vector<int, AllocWithCopyPropagation<int>> D5(10, alloc_copy_prop2);
    //since there is copy propagation, we destroy all 10 elements, and use the propagated allocator to construct
    // 5 new elements
    D5 = D4; 
    if (!verify(D5, N, V)) {
        std::cout<<"Failed copy assign of AllocWithCopyProp"<<std::endl;
        return 1;
    }

    if (num_constructed_prop != 20 && num_destroyed_prop != 10)
    {
        std::cout<<"Allocator with copy propagation is copying incorrectly: ["<<num_constructed_prop<<", "<<num_destroyed_prop<<"]"<<std::endl;
        return 1;
    }
}