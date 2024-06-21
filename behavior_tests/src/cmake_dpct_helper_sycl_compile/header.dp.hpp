#ifndef HEADERADDED
#define HEADERADDED
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void add_wrapper(int, float *, float *, int, int, sycl::queue &);
EXPORT void product_wrapper(int, float *, float *, int, int, sycl::queue &);
EXPORT void subtract_wrapper(int, float *, float *, int, int, sycl::queue &);
EXPORT void twice_wrapper(int, float *, int, int, sycl::queue &);
#endif
