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

// adding these global variables to track construction and destruction
//  using custom allocators with different settings.
static int num_constructed_prop = 0;
static int num_destroyed_prop = 0;
static int num_constructed_no_prop = 0;
static int num_destroyed_no_prop = 0;

template <typename T>
class AllocWithNoMovePropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
  public:
  AllocWithNoMovePropagation(const sycl::context &Ctxt, const sycl::device &Dev,
                                     const sycl::property_list &PropList = {}) 
                       : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
  {}
  AllocWithNoMovePropagation(const sycl::queue &Q, const sycl::property_list &PropList = {}) 
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
  {}
  
  AllocWithNoMovePropagation(const AllocWithNoMovePropagation& other)
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
  {}

  typedef ::std::false_type propagate_on_container_move_assignment;
  static void construct(T *p) { ::new((void*)p) T();  num_constructed_no_prop++;}
  template <typename _Arg>
  static void construct(T *p, _Arg arg) { ::new((void*)p) T(arg); num_constructed_no_prop++;}
  static void destroy(T *p) { p->~T(); num_destroyed_no_prop++;}
  
};

template <typename T>
class AllocWithMovePropagation : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
  public:
  AllocWithMovePropagation(const sycl::context &Ctxt, const sycl::device &Dev,
                                     const sycl::property_list &PropList = {})
                       : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Ctxt, Dev, PropList)
  {}
  AllocWithMovePropagation(const sycl::queue &Q, const sycl::property_list &PropList = {}) 
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(Q, PropList)
  {}
  
  AllocWithMovePropagation(const AllocWithMovePropagation& other)
    : sycl::usm_allocator<T, sycl::usm::alloc::shared>(other)
  {}

  typedef ::std::true_type propagate_on_container_move_assignment;
  static void construct(T *p) { ::new((void*)p) T();  num_constructed_prop++;}
  template <typename _Arg>
  static void construct(T *p, _Arg arg) { ::new((void*)p) T(arg); num_constructed_prop++;}
  static void destroy(T *p) { p->~T(); num_destroyed_prop++;}
};


int main(void) {
  constexpr int N = 4;
  constexpr int V = 42;
  // Construct D1 from move constructor.
  dpct::device_vector<int> D1(std::move(dpct::device_vector<int>(N, V)));
  if (!verify(D1, N, V)) {
    return 1;
  }
  // Move assign to D2.
  dpct::device_vector<int> D2;
  auto alloc = D2.get_allocator();
  D2 = std::move(dpct::device_vector<int>(N, V));
  if (!verify(D2, N, V)) {
    return 1;
  }

  // check appropriate effect of Allocator::propagate_on_container_move_assignment
  AllocWithNoMovePropagation<int> alloc_no_move_prop(dpct::get_default_queue());
  
  dpct::device_vector<int, AllocWithNoMovePropagation<int>> D3(std::move(D2), alloc_no_move_prop);
  if (!verify(D3, N, V)) {
    std::cout<<"Failed move of default allocator to AllocWithNoMoveProp"<<std::endl;
    return 1;
  }
  dpct::device_vector<int, AllocWithNoMovePropagation<int>> D4(std::move(D3));
  if (!verify(D4, N, V)) {
    std::cout<<"Failed move of AllocWithNoMoveProp"<<std::endl;
    return 1;
  }
  dpct::device_vector<int, AllocWithNoMovePropagation<int>> D5;
  D5 = std::move(D4);
  if (!verify(D5, N, V)) {
    std::cout<<"Failed move assign of AllocWithNoMoveProp"<<std::endl;
    return 1;
  }

  if (num_constructed_no_prop != 8 && num_destroyed_no_prop != 4)
  {
    std::cout<<"Allocator without move propagation is moving incorrectly: ["
             <<num_constructed_no_prop<<", "<<num_destroyed_no_prop<<"]"<<std::endl;
    return 1;
  }

  AllocWithMovePropagation<int> alloc_move_prop(dpct::get_default_queue());

  dpct::device_vector<int, AllocWithMovePropagation<int>> D6(std::move(D5), alloc_move_prop);
  if (!verify(D6, N, V)) {
    std::cout<<"Failed move of AllocWithNoMoveProp to AllocWithMoveProp"<<std::endl;
    return 1;
  }
  dpct::device_vector<int, AllocWithMovePropagation<int>> D7;
  D7 = std::move(D6);
  if (!verify(D7, N, V)) {
    std::cout<<"Failed move assign of AllocWithMoveProp"<<std::endl;
    return 1;
  }
  dpct::device_vector<int, AllocWithMovePropagation<int>> D8(std::move(D7));
  if (!verify(D8, N, V)) {
    std::cout<<"Failed move of AllocWithMoveProp"<<std::endl;
    return 1;
  }
  if (num_constructed_prop != 4 && num_destroyed_prop != 0)
  {
    std::cout<<"Allocator with move propagation is moving incorrectly: ["
             <<num_constructed_prop<<", "<<num_destroyed_prop<<"]"<<std::endl;
    return 1;
  }

  return 0;
}
