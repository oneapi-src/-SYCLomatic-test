// ===------- util_shift_sub_group_left_extension.cpp------ -*- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <algorithm>
#include <dpct/dpct.hpp>
#include <iterator>
#include <sycl/sycl.hpp>

enum class TestKind { Up, Down };

template <TestKind Kind, int LogicWarpSize>
void shuffle(int *data, unsigned src, unsigned bound, unsigned mask,
             const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_sub_group().get_local_linear_id();
  if constexpr (Kind == TestKind::Up)
    data[tid] = dpct::experimental::shift_sub_group_right<LogicWarpSize>(
        item_ct1.get_sub_group(), tid, src, bound, mask);
  else
    data[tid] = dpct::experimental::shift_sub_group_left<LogicWarpSize>(
        item_ct1.get_sub_group(), tid, src, bound, mask);
}

template <int LogicWarpSize, int N> void print_array(const int (&arr)[N]) {
#define PRINT_IMPL(VAL)                                                        \
  for (int i = 0; i < 32 / LogicWarpSize; ++i) {                               \
    printf("[");                                                               \
    for (int j = 0; j < LogicWarpSize; ++j) {                                  \
      int idx = i * LogicWarpSize + j;                                         \
      printf("%2d%s", VAL, (idx == LogicWarpSize - 1 ? "" : " "));             \
    }                                                                          \
    printf("]");                                                               \
  }                                                                            \
  printf("\n");
  PRINT_IMPL(idx);
  PRINT_IMPL(arr[idx]);
#undef PRINT_IMPL
}

template <TestKind Kind, int LogicWarpSize>
bool test_shuffle(const int (&exp)[32], unsigned src, unsigned bound,
                  unsigned mask) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  assert(32 % LogicWarpSize == 0);
  int *d_data, h_data[32];
  d_data = (int *)sycl::malloc_device(sizeof(h_data), q_ct1);
  q_ct1.memset(d_data, 0, sizeof(h_data)).wait();
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        shuffle<Kind, LogicWarpSize>(d_data, src, bound, mask, item_ct1);
      });
  q_ct1.wait();
  q_ct1.memcpy(h_data, d_data, sizeof(h_data)).wait();
  dpct::dpct_free(d_data, q_ct1);
  printf("Test: [Kind = %s, src = %u, bound = %u, mask = 0x%x] ",
         Kind == TestKind::Up ? "ShuffleUp" : "ShuffleDown", src, bound, mask);
  if (!std::equal(std::begin(h_data), std::end(h_data), std::begin(exp))) {
    printf("FAIL\n");
    printf("Expected:\n");
    print_array<LogicWarpSize>(exp);
    printf("But got:\n");
    print_array<LogicWarpSize>(h_data);
    return false;
  }
  printf("PASS\n");
  return true;
}

int main() {
  test_shuffle<TestKind::Up, 8>({0,  1,  2,  3,  0,  1,  2,  3,  8,  9,  10,
                                 11, 8,  9,  10, 11, 16, 17, 18, 19, 16, 17,
                                 18, 19, 24, 25, 26, 27, 24, 25, 26, 27},
                                4, 0, 0xFFFFFFFF);
  test_shuffle<TestKind::Up, 8>({0,  1,  2,  3,  0,  1,  2,  3,  8,  9,  10,
                                 11, 8,  9,  10, 11, 16, 17, 18, 19, 16, 17,
                                 18, 19, 24, 25, 26, 27, 24, 25, 26, 27},
                                4, 0, 0x0);
  test_shuffle<TestKind::Up, 8>({0,  1,  2,  3,  0,  1,  2,  3,  8,  9,  10,
                                 11, 8,  9,  10, 11, 16, 17, 18, 19, 16, 17,
                                 18, 19, 24, 25, 26, 27, 24, 25, 26, 27},
                                4, 0, 0xa);
  test_shuffle<TestKind::Up, 8>({0,  1,  2,  0,  1,  2,  3,  4,  8,  9,  10,
                                 8,  9,  10, 11, 12, 16, 17, 18, 16, 17, 18,
                                 19, 20, 24, 25, 26, 24, 25, 26, 27, 28},
                                3, 0, 0xFFFFFFFF);
  test_shuffle<TestKind::Up, 4>({0,  1,  0,  1,  4,  5,  4,  5,  8,  9,  8,
                                 9,  12, 13, 12, 13, 16, 17, 16, 17, 20, 21,
                                 20, 21, 24, 25, 24, 25, 28, 29, 28, 29},
                                2, 0, 0xFFFFFFFF);
  test_shuffle<TestKind::Up, 4>({0,  1,  0,  1,  4,  5,  4,  5,  8,  9,  8,
                                 9,  12, 13, 12, 13, 16, 17, 16, 17, 20, 21,
                                 20, 21, 24, 25, 24, 25, 28, 29, 28, 29},
                                2, 0, 0xAAAAAAAA);
  test_shuffle<TestKind::Down, 8>({4,  5,  6,  7,  4,  5,  6,  7,  12, 13, 14,
                                   15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21,
                                   22, 23, 28, 29, 30, 31, 28, 29, 30, 31},
                                  4, 7, 0xFFFFFFFF);
  test_shuffle<TestKind::Down, 8>({4,  5,  6,  7,  4,  5,  6,  7,  12, 13, 14,
                                   15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21,
                                   22, 23, 28, 29, 30, 31, 28, 29, 30, 31},
                                  4, 7, 0x0);
  test_shuffle<TestKind::Down, 8>({4,  5,  6,  7,  4,  5,  6,  7,  12, 13, 14,
                                   15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21,
                                   22, 23, 28, 29, 30, 31, 28, 29, 30, 31},
                                  4, 7, 0xa);
  test_shuffle<TestKind::Down, 8>({4,  5,  2,  3,  4,  5,  6,  7,  12, 13, 10,
                                   11, 12, 13, 14, 15, 20, 21, 18, 19, 20, 21,
                                   22, 23, 28, 29, 26, 27, 28, 29, 30, 31},
                                  4, 5, 0xFFFFFFFF);
  test_shuffle<TestKind::Down, 4>({2,  3,  2,  3,  6,  7,  6,  7,  10, 11, 10,
                                   11, 14, 15, 14, 15, 18, 19, 18, 19, 22, 23,
                                   22, 23, 26, 27, 26, 27, 30, 31, 30, 31},
                                  2, 3, 0xFFFFFFFF);
  test_shuffle<TestKind::Down, 4>({2,  3,  2,  3,  6,  7,  6,  7,  10, 11, 10,
                                   11, 14, 15, 14, 15, 18, 19, 18, 19, 22, 23,
                                   22, 23, 26, 27, 26, 27, 30, 31, 30, 31},
                                  2, 3, 0xAAAAAAAA);
  return 0;
}
