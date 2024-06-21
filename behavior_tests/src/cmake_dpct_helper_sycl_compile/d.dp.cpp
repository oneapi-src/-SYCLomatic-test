#include "header.dp.hpp"

void twice(int size, float *res, const sycl::nd_item<3> &item_ct1) {
  int blockIndex = item_ct1.get_group(2);
  int blockSize = item_ct1.get_local_range(2);
  int threadIndex = item_ct1.get_local_id(2);
  int index = blockIndex * blockSize + threadIndex;
  res[index] += res[index];
}

void twice_wrapper(int size, float *d, int numOfBlocks, int threadsPerBlock,
                   sycl::queue &q_ct1) {
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) { twice(size, d, item_ct1); });
}
