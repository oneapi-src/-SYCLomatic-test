#include <cub/cub.cuh>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc<T>(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

bool test_arg_max() {
  int num_items = 7;
  int *d_in = init({8, 6, 7, 5, 3, 0, 9});
  cub::KeyValuePair<int, int> *d_out =
                                  init<cub::KeyValuePair<int, int>>({{-1, -1}}),
                              out;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  return out.key == 6 && out.value == 9;
}

bool test_arg_max_non_defaule_stream() {
  int num_items = 7;
  int *d_in = init({8, 6, 7, 5, 3, 0, 9});
  cub::KeyValuePair<int, int> *d_out =
                                  init<cub::KeyValuePair<int, int>>({{-1, -1}}),
                              out;
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items, s);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  cudaStreamDestroy(s);
  return out.key == 6 && out.value == 9;
}

bool test_arg_min() {
  int num_items = 7;
  int *d_in = init({8, 6, 7, 5, 3, 0, 9});
  cub::KeyValuePair<int, int> *d_out =
                                  init<cub::KeyValuePair<int, int>>({{-1, -1}}),
                              out;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  return out.key == 5 && out.value == 0;
}

bool test_arg_min_non_default_stream() {
  int num_items = 7;
  int *d_in = init({8, 6, 7, 5, 3, 0, 9});
  cub::KeyValuePair<int, int> *d_out =
                                  init<cub::KeyValuePair<int, int>>({{-1, -1}}),
                              out;
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            num_items, s);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  cudaStreamDestroy(s);
  return out.key == 5 && out.value == 0;
}

__global__ void test_arg_min_max_op_in_device(int *Res) {
  cub::KeyValuePair<int, int> LHS{1, 3}, RHS{2, 4};
  cub::ArgMin MinOp;
  cub::ArgMax MaxOp;
  auto Min = MinOp(LHS, RHS);
  auto Max = MaxOp(LHS, RHS);
  *Res = Min.key == 1 && Min.value == 3 && Max.key == 2 && Max.value == 4;
}

bool test_arg_min_max_op_in_host() {
  cub::KeyValuePair<int, int> LHS{1, 3}, RHS{2, 4};
  cub::ArgMin MinOp;
  cub::ArgMax MaxOp;
  auto Min = MinOp(LHS, RHS);
  auto Max = MaxOp(LHS, RHS);
  return Min.key == 1 && Min.value == 3 && Max.key == 2 && Max.value == 4;
}

bool test_arg_min_max_op() {
  int *Res;
  cudaMallocManaged(&Res, sizeof(int));
  *Res = 0;
  test_arg_min_max_op_in_device<<<1, 1>>>(Res);
  cudaDeviceSynchronize();
  bool Val = *Res;
  cudaFree(Res);
  return Val && test_arg_min_max_op_in_host();
}

int main() {
  int res = 0;
  if (!test_arg_max()) {
    res = 1;
    std::cout << "cub::DeviceReduce::ArgMax test failed\n";
  }

  if (!test_arg_max_non_defaule_stream()) {
    res = 1;
    std::cout << "cub::DeviceReduce::ArgMax(Non default stream) test failed\n";
  }

  if (!test_arg_min()) {
    res = 1;
    std::cout << "cub::DeviceReduce::ArgMin test failed\n";
  }

  if (!test_arg_min_non_default_stream()) {
    res = 1;
    std::cout << "cub::DeviceReduce::ArgMin(Non default stream) test failed\n";
  }

  if (!test_arg_min_max_op()) {
    res = 1;
    std::cout << "cub::{ArgMin, ArgMax} binary operator test failed\n";
  }

  return res;
}
