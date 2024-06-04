// ====------ cub_device_spmv.cu------------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdlib>
#include <cub/cub.cuh>
#include <initializer_list>
#include <iterator>

namespace detail {

template <typename Callable> class scope_exit {
  Callable ExitFunction;

public:
  template <typename Fp>
  explicit scope_exit(Fp &&F) : ExitFunction(std::forward<Fp>(F)) {}

  scope_exit(const scope_exit &) = delete;
  scope_exit &operator=(scope_exit &&) = delete;
  scope_exit &operator=(const scope_exit &) = delete;

  ~scope_exit() { ExitFunction(); }
};
} // end namespace detail

template <typename Callable>
[[nodiscard]] detail::scope_exit<std::decay_t<Callable>>
make_scope_exit(Callable &&F) {
  return detail::scope_exit<std::decay_t<Callable>>(std::forward<Callable>(F));
}

template <class T> T *init(std::initializer_list<T> L) {
  T *Ptr = nullptr;
  cudaMallocManaged(&Ptr, sizeof(T) * L.size());
  std::copy(L.begin(), L.end(), Ptr);
  return Ptr;
}

int main() {
  int num_rows = 9;
  int num_cols = 9;
  int num_nonzeros = 24;
  float *d_values = init<float>(
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  int *d_column_indices = init(
      {1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7});

  int *d_row_offsets = init({0, 2, 5, 7, 10, 14, 17, 19, 22, 24});

  float *d_vector_x = init<float>({1, 1, 1, 1, 1, 1, 1, 1, 1});
  float *d_vector_y = init<float>({0, 1, 0, 0, 0, 0, 0, 0, 0});

  [[maybe_unused]] auto _ = make_scope_exit([&]() {
    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);
    cudaFree(d_vector_x);
    cudaFree(d_vector_y);
  });

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros);

  cudaDeviceSynchronize();

  float expected[] = {2, 3, 2, 3, 4, 3, 2, 3, 2};
  if (!std::equal(std::begin(expected), std::end(expected), d_vector_y)) {
    std::cout << "cub::DeviceSpmv::CsrMV FAIL\n";
    return 1;
  }
  std::cout << "cub::DeviceSpmv::CsrMV PASS\n";
  return 0;
}
