// ===------------------- math.cpp ---------- -*- C++ -* ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <dpct/dpct.hpp>

int passed = 0;
int failed = 0;

template <typename T>
void check(const std::string &name, const T &expect, const T &output) {
  printf("%s: ", name.c_str());
  if constexpr (std::is_integral_v<T>) {
    printf("(%d==%d) -- ", expect, output);
    if (expect == output) {
      printf("pass\n");
      ++passed;
      return;
    }
  } else {
    printf("(%f==%f) -- ", expect, output);
    if (expect > output - 0.0001 && expect < output + 0.0001) {
      printf("pass\n");
      ++passed;
      return;
    }
  }
  ++failed;
  printf("failed\n");
}

int main() {
  double d;
  float f;
  int i;
  dpct::fast_length(&f, i);
  dpct::length(&d, i);
  check("clamp", 4, dpct::clamp(5, 3, 4));
  check("clamp", 4.0, dpct::clamp(5.0, 3.0, 4.0));
  check("compare", true,
        dpct::compare(sycl::half(1), sycl::half(1), std::equal_to<>()));
  check(
      "unordered_compare", true,
      dpct::unordered_compare(sycl::half(1), sycl::half(1), std::equal_to<>()));
  check("compare_both", false,
        dpct::compare_both(sycl::half2(1, 2), sycl::half2(2, 1),
                           std::equal_to<>()));
  check("unordered_compare_both", false,
        dpct::unordered_compare_both(sycl::half2(1, 2), sycl::half2(2, 1),
                                     std::equal_to<>()));
  check("compare", 0.0,
        (double)dpct::compare(sycl::half2(1, 2), sycl::half2(2, 1),
                              std::equal_to<>())[0]);
  check("compare_mask", (unsigned)0,
        dpct::compare_mask(sycl::half2(1, 2), sycl::half2(2, 1),
                           std::equal_to<>()));
  check(
      "unordered_compare", true,
      dpct::unordered_compare(sycl::half(1), sycl::half(1), std::equal_to<>()));
  check("unordered_compare_mask", (unsigned)0,
        dpct::unordered_compare_mask(sycl::half2(1, 2), sycl::half2(2, 1),
                                     std::equal_to<>()));
  check("isnan", 0.0, (double)dpct::isnan(sycl::half2(1, 2))[0]);
  check("cbrt", 1.0, dpct::cbrt(1.0));
  check("relu", 1.0, dpct::relu(1.0));
  check("relu", 1.0, (double)dpct::relu(sycl::half2(1, 2))[0]);
  check("complex_mul_add", -2.0,
        (double)dpct::complex_mul_add(sycl::half2(1, 2), sycl::half2(1, 2),
                                      sycl::half2(1, 2))[0]);
  check("fmax_nan", 2.0, dpct::fmax_nan(1.0, 2.0));
  check("fmax_nan", 1.0,
        (double)dpct::fmax_nan(sycl::half2(1, 2), sycl::half2(1, 2))[0]);
  check("fmin_nan", 1.0, dpct::fmin_nan(1.0, 2.0));
  check("fmin_nan", 1.0,
        (double)dpct::fmin_nan(sycl::half2(1, 2), sycl::half2(1, 2))[0]);
  check("vectorized_binary", (unsigned)1,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::abs_diff()));
  check("vectorized_binary", (unsigned)3,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::add_sat()));
  check("vectorized_binary", (unsigned)2,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::rhadd()));
  check("vectorized_binary", (unsigned)1,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::hadd()));
  check("vectorized_binary", (unsigned)2,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::maximum()));
  check("vectorized_binary", (unsigned)1,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::minimum()));
  check("vectorized_binary", (unsigned)65535,
        dpct::vectorized_binary<sycl::short2>(1, 2, dpct::sub_sat()));
  check("vectorized_isgreater", 0,
        dpct::vectorized_isgreater<sycl::short2>(1, 2));
  check("vectorized_max", 2, dpct::vectorized_max<sycl::short2>(1, 2));
  check("vectorized_min", 1, dpct::vectorized_min<sycl::short2>(1, 2));
  check("vectorized_unary", (unsigned)1,
        dpct::vectorized_unary<sycl::short2>(1, dpct::abs()));
  check("vectorized_sum_abs_diff", (unsigned)1,
        dpct::vectorized_sum_abs_diff<sycl::short2>(1, 2));
  printf("passed %d/%d cases!\n", passed, passed + failed);
  if (failed) {
    printf("failed!\n");
  }
  return failed;
}
