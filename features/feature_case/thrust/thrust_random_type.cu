// ====------ thrust_random_type.cu--------------- *- CUDA -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>

struct random_1 {
  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(1.0f, 2.0f);
    rng.discard(n);

    return dist(rng);
  }
};

struct random_2 {
  __device__ float operator()(const unsigned int n) {
    thrust::default_random_engine rng;
    rng.discard(n);
    return (float)rng() / thrust::default_random_engine::max;
  }
};

int main(void) {
  {
    const int N = 20;
    thrust::device_vector<float> numbers(N);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin, index_sequence_begin + N,
                      numbers.begin(), random_1());

    for (int i = 0; i < N; i++) {
      if (numbers[i] > 2.0f || numbers[i] < 1.0f) {
        std::cout << "Test1 failed\n";
        return -1;
      }
    }

    thrust::transform(index_sequence_begin, index_sequence_begin + N,
                      numbers.begin(), random_2());
    for (int i = 0; i < N; i++) {
      if (numbers[i] > 1.0f || numbers[i] < 0.0f) {
        std::cout << "Test2 failed\n";
        return -1;
      }
    }
  }

  {

    // create a uniform_int_distribution to produce ints from [-5,10]
    thrust::uniform_int_distribution<int> dist(-5, 10);
    if (dist.min() != -5 || dist.max() < 10 || dist.a() != -5 ||
        dist.b() != 10) {
      std::cout << "Test3 failed\n";
      return -1;
    }
  }

  {
    // create a normal_distribution to produce floats from the Normal
    // distribution with mean 1.0 and standard deviation 2.0
    thrust::normal_distribution<float> dist(1.0f, 2.0f);

    if (dist.mean() != 1.0f || dist.stddev() != 2.0f) {
      std::cout << "Test4 failed\n";
      return -1;
    }
  }

  {
    thrust::host_vector<thrust::complex<float>> h_complex(4);
    h_complex[0] = thrust::complex<float>(1.0, 1.0);  // 1 + 1i
    h_complex[1] = thrust::complex<float>(0.0, 1.0);  // 0 + 1i
    h_complex[2] = thrust::complex<float>(-1.0, 0.0); // -1 + 0i
    h_complex[3] = thrust::complex<float>(0.0, -1.0); // 0 - 1i

    // Copy host vector to device vector
    thrust::device_vector<thrust::complex<float>> d_complex = h_complex;

    // Create a device vector to store the results
    thrust::device_vector<float> d_results(4);

    // Compute arguments (angles) of complex numbers
    thrust::transform(
        d_complex.begin(), d_complex.end(), d_results.begin(),
        [] __device__(thrust::complex<float> z) { return thrust::arg(z); });

    // Copy results back to host
    thrust::host_vector<float> h_results = d_results;

    float ref[4] = {0.785398, 1.5708, 3.14159, -1.5708};

    for (int i = 0; i < 4; i++) {
      if (std::fabs(ref[i] - h_results[i]) > 1e-5) {
        std::cout << "Test5 failed\n";
        return -1;
      }
    }
  }

  std::cout << "Test passed\n";
  return 0;
}
