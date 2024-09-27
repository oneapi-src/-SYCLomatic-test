// ====--------------- rng_host.cpp---------- -*- C++ -* ----------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>

#include <cstdio>

using namespace dpct::rng;

int ret = 0;
const size_t num = 10;

template <typename num_type>
void check(const std::string &s, num_type *output_0, num_type *output_1) {
  for (size_t i = 0; i < num; ++i) {
    if (output_0[i] != output_1[i]) {
      std::cout << s << " failed!" << std::endl;
      std::cout << "output_0 is: ";
      for (size_t i = 0; i < num; ++i) {
        std::cout << output_0[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "output_1 is: ";
      for (size_t i = 0; i < num; ++i) {
        std::cout << output_1[i] << " ";
      }
      std::cout << std::endl;
      ret++;
      return;
    }
  }
  std::cout << s << " passed!" << std::endl;
}

template <typename engine_type>
void test_engine(host_rng_ptr rng0, engine_type &rng1) {
  unsigned int *output_i_0 =
      sycl::malloc_host<unsigned int>(num, dpct::get_default_queue());
  unsigned int *output_i_1 =
      sycl::malloc_host<unsigned int>(num, dpct::get_default_queue());
  unsigned long long *output_ll_0 =
      sycl::malloc_host<unsigned long long>(num, dpct::get_default_queue());
  unsigned long long *output_ll_1 =
      sycl::malloc_host<unsigned long long>(num, dpct::get_default_queue());
  float *output_f_0 = sycl::malloc_host<float>(num, dpct::get_default_queue());
  float *output_f_1 = sycl::malloc_host<float>(num, dpct::get_default_queue());
  double *output_d_0 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());
  double *output_d_1 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());

  if (!std::is_same_v<engine_type, oneapi::mkl::rng::mrg32k3a> &&
      !std::is_same_v<engine_type, oneapi::mkl::rng::sobol>) {
    rng0->generate_uniform_bits(output_i_0, num);
    oneapi::mkl::rng::generate(
        oneapi::mkl::rng::uniform_bits<std::uint32_t>(), rng1, num,
        dpct::detail::get_memory<std::uint32_t>((void *)output_i_1));
    check(std::string(typeid(engine_type).name()) + "::generate_uniform_bits_i",
          output_i_0, output_i_1);

    rng0->generate_uniform_bits(output_ll_0, num);
    oneapi::mkl::rng::generate(
        oneapi::mkl::rng::uniform_bits<std::uint64_t>(), rng1, num,
        dpct::detail::get_memory<std::uint64_t>((void *)output_ll_1));
    check(std::string(typeid(engine_type).name()) +
              "::generate_uniform_bits_ll",
          output_ll_0, output_ll_1);
  }

  rng0->generate_lognormal(output_f_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::lognormal<float>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<float>(output_f_1));
  check(std::string(typeid(engine_type).name()) + "::generate_lognormal_f",
        output_f_0, output_f_1);

  rng0->generate_lognormal(output_d_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::lognormal<double>(0.5, 0.5),
                             rng1, num, dpct::detail::get_memory<double>(output_d_1));
  check(std::string(typeid(engine_type).name()) + "::generate_lognormal_d",
        output_d_0, output_d_1);

  rng0->generate_gaussian(output_f_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<float>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<float>(output_f_1));
  check(std::string(typeid(engine_type).name()) + "::generate_gaussian_f",
        output_f_0, output_f_1);

  rng0->generate_gaussian(output_d_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<double>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<double>(output_d_1));
  check(std::string(typeid(engine_type).name()) + "::generate_gaussian_d",
        output_d_0, output_d_1);

  rng0->generate_poisson(output_i_0, num, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::poisson<unsigned int>(0.5), rng1,
                             num, dpct::detail::get_memory<unsigned int>(output_i_1));
  check(std::string(typeid(engine_type).name()) + "::generate_poisson",
        output_i_0, output_i_1);

  rng0->generate_uniform(output_f_0, num);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<float>(), rng1, num,
                             dpct::detail::get_memory<float>(output_f_1));
  check(std::string(typeid(engine_type).name()) + "::generate_uniform_f",
        output_f_0, output_f_1);

  rng0->generate_uniform(output_d_0, num);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<double>(), rng1, num,
                             dpct::detail::get_memory<double>(output_d_1));
  check(std::string(typeid(engine_type).name()) + "::generate_uniform_d",
        output_d_0, output_d_1);
}

template <typename engine_type>
void test_engine_with_skip(host_rng_ptr rng0, engine_type &rng1) {
  const std::uint64_t skip = 4;
  double *output_d_0 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());
  double *output_d_1 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());
  test_engine(rng0, rng1);
  rng0->skip_ahead(skip);
  oneapi::mkl::rng::skip_ahead(rng1, skip);
  rng0->generate_uniform(output_d_0, num);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<double>(), rng1, num,
                             dpct::detail::get_memory<double>(output_d_1));
  check(std::string(typeid(engine_type).name()) + "::skip_ahead", output_d_0,
        output_d_1);
}

void test_sobol_with_set_direction_numbers(host_rng_ptr rng0,
                                           oneapi::mkl::rng::sobol &rng1) {
  unsigned int *output_i_0 =
      sycl::malloc_host<unsigned int>(num, dpct::get_default_queue());
  unsigned int *output_i_1 =
      sycl::malloc_host<unsigned int>(num, dpct::get_default_queue());
  unsigned long long *output_ll_0 =
      sycl::malloc_host<unsigned long long>(num, dpct::get_default_queue());
  unsigned long long *output_ll_1 =
      sycl::malloc_host<unsigned long long>(num, dpct::get_default_queue());
  float *output_f_0 = sycl::malloc_host<float>(num, dpct::get_default_queue());
  float *output_f_1 = sycl::malloc_host<float>(num, dpct::get_default_queue());
  double *output_d_0 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());
  double *output_d_1 =
      sycl::malloc_host<double>(num, dpct::get_default_queue());

  rng0->generate_lognormal(output_f_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::lognormal<float>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<float>(output_f_1));
  check("oneapi::mkl::rng::sobol::generate_lognormal_f with "
        "set_direction_numbers",
        output_f_0, output_f_1);

  rng0->generate_lognormal(output_d_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::lognormal<double>(0.5, 0.5),
                             rng1, num, dpct::detail::get_memory<double>(output_d_1));
  check("oneapi::mkl::rng::sobol::generate_lognormal_d with "
        "set_direction_numbers",
        output_d_0, output_d_1);

  rng0->generate_gaussian(output_f_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<float>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<float>(output_f_1));
  check(
      "oneapi::mkl::rng::sobol::generate_gaussian_f with set_direction_numbers",
      output_f_0, output_f_1);

  rng0->generate_gaussian(output_d_0, num, 0.5, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<double>(0.5, 0.5), rng1,
                             num, dpct::detail::get_memory<double>(output_d_1));
  check(
      "oneapi::mkl::rng::sobol::generate_gaussian_d with set_direction_numbers",
      output_d_0, output_d_1);

  rng0->generate_poisson(output_i_0, num, 0.5);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::poisson<unsigned int>(0.5), rng1,
                             num, dpct::detail::get_memory<unsigned int>(output_i_1));
  check("oneapi::mkl::rng::sobol::generate_poisson with set_direction_numbers",
        output_i_0, output_i_1);

  rng0->generate_uniform(output_f_0, num);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<float>(), rng1, num,
                             dpct::detail::get_memory<float>(output_f_1));
  check(
      "oneapi::mkl::rng::sobol::generate_uniform_f with set_direction_numbers",
      output_f_0, output_f_1);

  rng0->generate_uniform(output_d_0, num);
  oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<double>(), rng1, num,
                             dpct::detail::get_memory<double>(output_d_1));
  check(
      "oneapi::mkl::rng::sobol::generate_uniform_d with set_direction_numbers",
      output_d_0, output_d_1);
}

int main() {
  const std::uint64_t seed = 1;
  const std::uint32_t dimensions = 1;
  const std::uint64_t skip = 4;
  sycl::queue q;
  host_rng_ptr rng0;

  rng0 = create_host_rng(random_engine_type::philox4x32x10);
  rng0->set_seed(seed);
  rng0->set_queue(&q);
  oneapi::mkl::rng::philox4x32x10 rng1(q, seed);
  test_engine_with_skip(rng0, rng1);

  rng0 = create_host_rng(random_engine_type::mrg32k3a);
  rng0->set_seed(seed);
  rng0->set_queue(&q);
  oneapi::mkl::rng::mrg32k3a rng2(q, seed);
  test_engine_with_skip(rng0, rng2);

  rng0 = create_host_rng(random_engine_type::mt2203);
  rng0->set_seed(seed);
  rng0->set_queue(&q);
  oneapi::mkl::rng::mt2203 rng3(q, seed);
  test_engine(rng0, rng3);

  rng0 = create_host_rng(random_engine_type::mt19937);
  rng0->set_seed(seed);
  rng0->set_queue(&q);
  oneapi::mkl::rng::mt19937 rng4(q, seed);
  test_engine_with_skip(rng0, rng4);

  rng0 = create_host_rng(random_engine_type::sobol);
  rng0->set_dimensions(dimensions);
  rng0->set_queue(&q);
  oneapi::mkl::rng::sobol rng5(q, dimensions);
  test_engine_with_skip(rng0, rng5);

  std::vector<std::uint32_t> vec = {2, 3, 5, 7};
  rng0 = create_host_rng(random_engine_type::sobol);
  rng0->set_direction_numbers(vec);
  rng0->set_queue(&q);
  oneapi::mkl::rng::sobol rng6(q, vec);
  test_sobol_with_set_direction_numbers(rng0, rng6);

  std::cout << "ret = " << ret << std::endl;
  return 0;
}
