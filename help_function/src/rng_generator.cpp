// ====------ rng_generator.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <dpct/rng_utils.hpp>
#include <cstdio>
#include <math.h>

template<class TO, class FROM>
void copy4(TO *to, FROM from) {
  to[0] = from.x();
  to[1] = from.y();
  to[2] = from.z();
  to[3] = from.w();
}

template<class TO, class FROM>
void copy2(TO *to, FROM from) {
  to[0] = from.x();
  to[1] = from.y();
}

void test_bits(unsigned int *u) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  u[0] = rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
  u[1] = rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
  u[2] = rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
  u[3] = rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
  sycl::uint2 u2_1 =
      rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 2>();
  sycl::uint2 u2_2 =
      rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 2>();
  sycl::uint4 u4_1 =
      rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 4>();
  copy2(u+4, u2_1);
  copy2(u+6, u2_2);
  copy4(u+8, u4_1);
}

void ref_bits(unsigned int *u) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::bits<std::uint32_t> distr;
  sycl::uint4 u4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::uint4 u4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::uint4 u4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(u, u4_1);
  copy4(u+4, u4_2);
  copy4(u+8, u4_3);
}

void test_normal(float *f) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  f[0] = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
  f[1] = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
  f[2] = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
  f[3] = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
  sycl::float2 f2_1 =
      rng.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
  sycl::float2 f2_2 =
      rng.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
  sycl::float4 f4_1 =
      rng.generate<oneapi::mkl::rng::device::gaussian<float>, 4>();
  copy2(f+4, f2_1);
  copy2(f+6, f2_2);
  copy4(f+8, f4_1);
}

void ref_normal(float *f) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::gaussian<float> distr;
  sycl::float4 f4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(f, f4_1);
  copy4(f+4, f4_2);
  copy4(f+8, f4_3);
}

void test_normal_double(double *d) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  d[0] = rng.generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  d[1] = rng.generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  d[2] = rng.generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  d[3] = rng.generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  sycl::double2 d2_1 =
      rng.generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
  sycl::double2 d2_2 =
      rng.generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
  sycl::double4 d4_1 =
      rng.generate<oneapi::mkl::rng::device::gaussian<double>, 4>();
  copy2(d+4, d2_1);
  copy2(d+6, d2_2);
  copy4(d+8, d4_1);
}

void ref_normal_double(double *d) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::gaussian<double> distr;
  sycl::double4 d4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(d, d4_1);
  copy4(d+4, d4_2);
  copy4(d+8, d4_3);
}

void test_log_normal(float *f) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  f[0] = rng.generate<oneapi::mkl::rng::device::lognormal<float>, 1>(3, 7);
  f[1] = rng.generate<oneapi::mkl::rng::device::lognormal<float>, 1>(3, 7);
  f[2] = rng.generate<oneapi::mkl::rng::device::lognormal<float>, 1>(3, 7);
  f[3] = rng.generate<oneapi::mkl::rng::device::lognormal<float>, 1>(3, 7);
  sycl::float2 f2_1 =
      rng.generate<oneapi::mkl::rng::device::lognormal<float>, 2>(3, 7);
  sycl::float2 f2_2 =
      rng.generate<oneapi::mkl::rng::device::lognormal<float>, 2>(3, 7);
  sycl::float4 f4_1 =
      rng.generate<oneapi::mkl::rng::device::lognormal<float>, 4>(3, 7);
  copy2(f+4, f2_1);
  copy2(f+6, f2_2);
  copy4(f+8, f4_1);
}

void ref_log_normal(float *f) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::lognormal<float> distr(3, 7);
  sycl::float4 f4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(f, f4_1);
  copy4(f+4, f4_2);
  copy4(f+8, f4_3);
}

void test_log_normal_double(double *d) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  d[0] = rng.generate<oneapi::mkl::rng::device::lognormal<double>, 1>(3, 7);
  d[1] = rng.generate<oneapi::mkl::rng::device::lognormal<double>, 1>(3, 7);
  d[2] = rng.generate<oneapi::mkl::rng::device::lognormal<double>, 1>(3, 7);
  d[3] = rng.generate<oneapi::mkl::rng::device::lognormal<double>, 1>(3, 7);
  sycl::double2 d2_1 =
      rng.generate<oneapi::mkl::rng::device::lognormal<double>, 2>(3, 7);
  sycl::double2 d2_2 =
      rng.generate<oneapi::mkl::rng::device::lognormal<double>, 2>(3, 7);
  sycl::double4 d4_1 =
      rng.generate<oneapi::mkl::rng::device::lognormal<double>, 4>(3, 7);
  copy2(d+4, d2_1);
  copy2(d+6, d2_2);
  copy4(d+8, d4_1);
}

void ref_log_normal_double(double *d) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::lognormal<double> distr(3, 7);
  sycl::double4 d4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(d, d4_1);
  copy4(d+4, d4_2);
  copy4(d+8, d4_3);
}

void test_uniform(float *f) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  f[0] = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
  f[1] = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
  f[2] = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
  f[3] = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
  sycl::float2 f2_1 =
      rng.generate<oneapi::mkl::rng::device::uniform<float>, 2>();
  sycl::float2 f2_2 =
      rng.generate<oneapi::mkl::rng::device::uniform<float>, 2>();
  sycl::float4 f4_1 =
      rng.generate<oneapi::mkl::rng::device::uniform<float>, 4>();
  copy2(f+4, f2_1);
  copy2(f+6, f2_2);
  copy4(f+8, f4_1);
}

void ref_uniform(float *f) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::uniform<float> distr;
  sycl::float4 f4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::float4 f4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(f, f4_1);
  copy4(f+4, f4_2);
  copy4(f+8, f4_3);
}

void test_uniform_double(double *d) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  d[0] = rng.generate<oneapi::mkl::rng::device::uniform<double>, 1>();
  d[1] = rng.generate<oneapi::mkl::rng::device::uniform<double>, 1>();
  d[2] = rng.generate<oneapi::mkl::rng::device::uniform<double>, 1>();
  d[3] = rng.generate<oneapi::mkl::rng::device::uniform<double>, 1>();
  sycl::double2 d2_1 =
      rng.generate<oneapi::mkl::rng::device::uniform<double>, 2>();
  sycl::double2 d2_2 =
      rng.generate<oneapi::mkl::rng::device::uniform<double>, 2>();
  sycl::double4 d4_1 =
      rng.generate<oneapi::mkl::rng::device::uniform<double>, 4>();
  copy2(d+4, d2_1);
  copy2(d+6, d2_2);
  copy4(d+8, d4_1);
}

void ref_uniform_double(double *d) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::uniform<double> distr;
  sycl::double4 d4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::double4 d4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(d, d4_1);
  copy4(d+4, d4_2);
  copy4(d+8, d4_3);
}

void test_poisson(unsigned int *u) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  u[0] = rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(3);
  u[1] = rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(3);
  u[2] = rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(3);
  u[3] = rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(3);
  sycl::uint2 u2_1 =
      rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 2>(3);
  sycl::uint2 u2_2 =
      rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 2>(3);
  sycl::uint4 u4_1 =
      rng.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 4>(3);
  copy2(u+4, u2_1);
  copy2(u+6, u2_2);
  copy4(u+8, u4_1);
}

void ref_poisson(unsigned int *u) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::poisson<std::uint32_t> distr(3);
  sycl::uint4 u4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::uint4 u4_2 = oneapi::mkl::rng::device::generate(distr, rng);
  sycl::uint4 u4_3 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(u, u4_1);
  copy4(u+4, u4_2);
  copy4(u+8, u4_3);
}

void test_skipahead(unsigned int *u) {
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      rng;
  rng = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(1, {3, 2 * 4});
  oneapi::mkl::rng::device::skip_ahead(rng.get_engine(), 7);
  sycl::uint4 u4_1 = rng.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 4>();
  copy4(u, u4_1);
}

void ref_skipahead(unsigned int *u) {
  oneapi::mkl::rng::device::philox4x32x10<4> rng(1, {3, 2 * 4});
  oneapi::mkl::rng::device::bits<std::uint32_t> distr;
  oneapi::mkl::rng::device::skip_ahead(rng, 7);
  sycl::uint4 u4_1 = oneapi::mkl::rng::device::generate(distr, rng);
  copy4(u, u4_1);
}

template<class T>
bool compare(T* data, T* ref) {
  bool res = true;
  for (size_t i = 0; i < 12; i++) {
    if ((fabs(data[i] - ref[i]) / ref[i]) > 0.01) {
      res = false;
      break;
    }
  }
  if (!res) {
    printf("data, ref:\n");
    for (size_t i = 0; i < 12; i++)
      printf("%.3f, %.3f\n", data[i], ref[i]);
  }
  return res;
}
template<>
bool compare(std::uint32_t* data, std::uint32_t* ref) {
  bool res = true;
  for (size_t i = 0; i < 12; i++) {
    if (data[i] != ref[i]) {
      res = false;
      break;
    }
  }
  if (!res) {
    printf("data, ref:\n");
    for (size_t i = 0; i < 12; i++)
      printf("%d, %d\n", data[i], ref[i]);
  }
  return res;
}

template<class T>
void clean_data(T* data, T* ref, sycl::queue &q) {
  q.memset(data, 0, sizeof(T) * 12).wait();
  q.memset(ref, 0, sizeof(T) * 12).wait();
}

int main() {
  bool result = true;
  sycl::queue q;
  float* f_d = sycl::malloc_shared<float>(12, q);
  double* d_d = sycl::malloc_shared<double>(12, q);
  unsigned int* u_d = sycl::malloc_shared<unsigned int>(12, q);
  float* f_ref = sycl::malloc_shared<float>(12, q);
  double* d_ref = sycl::malloc_shared<double>(12, q);
  unsigned int* u_ref = sycl::malloc_shared<unsigned int>(12, q);

  // bits
  clean_data(u_d, u_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_bits(u_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_bits(u_ref);
    });
  });
  q.wait();
  if (!compare(u_d, u_ref)) {
    std::cout << "bit fail" << std::endl;
    result = false;
  } else
    printf("bits pass\n");

  // normal
  clean_data(f_d, f_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_normal(f_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_normal(f_ref);
    });
  });
  q.wait();
  if (!compare(f_d, f_ref)) {
    std::cout << "normal fail" << std::endl;
    result = false;
  } else
    printf("normal pass\n");

  // normal double
  clean_data(d_d, d_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_normal_double(d_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_normal_double(d_ref);
    });
  });
  q.wait();
  if (!compare(d_d, d_ref)) {
    std::cout << "normal double fail" << std::endl;
    result = false;
  } else
    printf("normal double pass\n");

  // log_normal
  clean_data(f_d, f_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_log_normal(f_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_log_normal(f_ref);
    });
  });
  q.wait();
  if (!compare(f_d, f_ref)) {
    std::cout << "log_normal fail" << std::endl;
    result = false;
  } else
    printf("log_normal pass\n");

  // log normal double
  clean_data(d_d, d_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_log_normal_double(d_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_log_normal_double(d_ref);
    });
  });
  q.wait();
  if (!compare(d_d, d_ref)) {
    std::cout << "log_normal_double fail" << std::endl;
    result = false;
  } else
    printf("log_normal double pass\n");

  // uniform
  clean_data(f_d, f_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_uniform(f_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_uniform(f_ref);
    });
  });
  q.wait();
  if (!compare(f_d, f_ref)) {
    std::cout << "uniform fail" << std::endl;
    result = false;
  } else
    printf("uniform pass\n");

  // uniform double
  clean_data(d_d, d_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_uniform_double(d_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_uniform_double(d_ref);
    });
  });
  q.wait();
  if (!compare(d_d, d_ref)) {
    std::cout << "uniform double fail" << std::endl;
    result = false;
  } else
    printf("uniform double pass\n");

  // poisson
  clean_data(u_d, u_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_poisson(u_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_poisson(u_ref);
    });
  });
  q.wait();
  if (!compare(u_d, u_ref)) {
    std::cout << "poisson fail" << std::endl;
    result = false;
  } else
    printf("poisson pass\n");

  // skipahead
  clean_data(u_d, u_ref, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      test_skipahead(u_d);
    });
  });
  q.wait();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
      ref_skipahead(u_ref);
    });
  });
  q.wait();
  if (!compare(u_d, u_ref)) {
    std::cout << "skipahead fail" << std::endl;
    result = false;
  } else
    printf("skipahead pass\n");

  sycl::free(f_d, q);
  sycl::free(d_d, q);
  sycl::free(u_d, q);
  sycl::free(f_ref, q);
  sycl::free(d_ref, q);
  sycl::free(u_ref, q);

  if (result)
    return 0;
  return -1;
}
