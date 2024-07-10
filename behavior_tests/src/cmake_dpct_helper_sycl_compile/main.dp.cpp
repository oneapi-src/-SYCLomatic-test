#include "header.dp.hpp"

#include <iostream>
#include <math.h>

#define EXPECTED_VALUE 8.0f

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float *a, *b, *c, *d;
  int size = 1 << 20;

  a = sycl::malloc_shared<float>(size, q_ct1);
  b = sycl::malloc_shared<float>(size, q_ct1);
  c = sycl::malloc_shared<float>(size, q_ct1);
  d = sycl::malloc_shared<float>(size, q_ct1);

  for (int i = 0; i < size; ++i)
    a[i] = b[i] = c[i] = d[i] = 2.0f;

  int threadsPerBlock = 256;
  int numOfBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // eval 2.0 * 2.0
  product_wrapper(size, a, d, numOfBlocks, threadsPerBlock, q_ct1);
  dev_ct1.queues_wait_and_throw();

  // eval 4.0 + 2.0
  // The executable fails with missing symbol error at run time in Windows. This
  // seems to be icx-cl's bug. The following is the work-around for it.
  add_wrapper(size, a, d, numOfBlocks, threadsPerBlock, q_ct1);
  dev_ct1.queues_wait_and_throw();

  // eval 6.0 - 2.0
  subtract_wrapper(size, a, d, numOfBlocks, threadsPerBlock, q_ct1);
  dev_ct1.queues_wait_and_throw();

  // eval 4.0 + 4.0
  twice_wrapper(size, d, numOfBlocks, threadsPerBlock, q_ct1);
  dev_ct1.queues_wait_and_throw();

  float error = 0.0f;
  for (int i = 0; i < size; ++i)
    error = fmax(error, fabs(d[i] - EXPECTED_VALUE));

  sycl::free(a, q_ct1);
  sycl::free(b, q_ct1);
  sycl::free(c, q_ct1);
  sycl::free(d, q_ct1);

  // Expected value is "Error: 0"
  std::cout << "Error: " << error << "\n";

  return error == 0.0f ? 0 : 1;
}
