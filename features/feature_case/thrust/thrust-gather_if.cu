// ====------ thrust-gather.cu---------- *- CUDA -* ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

struct is_odd
{
  __host__ __device__
  bool operator()(const int x) const
  {
    return (x % 2) == 1;
  }
};

void test_1() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  thrust::device_vector<int> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);  
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10,123);

  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::device_vector<int> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10,123);
  thrust::gather_if(thrust::device, d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<int> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10,123);
  thrust::gather_if(thrust::seq, h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<int> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10,123);
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run passed!\n");
}


template<typename T>
void test_A() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  thrust::device_vector<T> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<T> d_output(10,123);

  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_A run failed\n");
      exit(-1);
    }
  }

  printf("test_A run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_B() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::device_vector<T> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);      
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<T> d_output(10,123);
  thrust::gather_if(thrust::device, d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_B run failed\n");
      exit(-1);
    }
  }
  printf("test_B run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_C() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<T> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);      
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<T> h_output(10,123);
  thrust::gather_if(thrust::seq, h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_C run failed\n");
      exit(-1);
    }
  }
  printf("test_C run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_D() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<T> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<T> h_output(10,123);
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_D run failed\n");
      exit(-1);
    }
  }
  printf("test_D run passed %ld!\n",sizeof(T));
}

void test_5() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::device_vector<int> d_values(values, values + 10);
  // select elements at even-indexed locations
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);
  // map all even indices into the first half of the range
  // and odd indices to the last half of the range
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10, 7);
  thrust::gather_if(thrust::device, d_map.begin(), d_map.end(),
                    d_stencil.begin(), d_values.begin(), d_output.begin());

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::device_vector<int> d_values(values, values + 10);
  // select elements at even-indexed locations
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);
  // map all even indices into the first half of the range
  // and odd indices to the last half of the range
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10, 7);
  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(),
                    d_values.begin(), d_output.begin());

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

void test_7() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::host_vector<int> h_values(values, values + 10);
  // select elements at even-indexed locations
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::host_vector<int> d_stencil(stencil, stencil + 10);
  // map all even indices into the first half of the range
  // and odd indices to the last half of the range
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::host_vector<int> d_map(map, map + 10);
  thrust::host_vector<int> d_output(10, 7);
  thrust::gather_if(thrust::host, d_map.begin(), d_map.end(), d_stencil.begin(),
                    h_values.begin(), d_output.begin());

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  printf("test_7 run passed!\n");
}

void test_8() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::host_vector<int> h_values(values, values + 10);
  // select elements at even-indexed locations
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::host_vector<int> d_stencil(stencil, stencil + 10);
  // map all even indices into the first half of the range
  // and odd indices to the last half of the range
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::host_vector<int> d_map(map, map + 10);
  thrust::host_vector<int> d_output(10, 7);
  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(),
                    h_values.begin(), d_output.begin());

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run passed!\n");
}

void test_9() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  // and odd indices to the last half of the range

  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

  int output[10] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  thrust::gather_if(thrust::host, map, map + 10, stencil, values, output);

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < 10; i++) {
    if (output[i] != values_ref[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  printf("test_9 run passed!\n");
}

void test_10() {

  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  // and odd indices to the last half of the range

  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

  int output[10] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  thrust::gather_if(map, map + 10, stencil, values, output);

  int values_ref[10] = {0, 7, 4, 7, 8, 7, 3, 7, 7, 7};
  for (int i = 0; i < 10; i++) {
    if (output[i] != values_ref[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  printf("test_10 run passed!\n");
}

void test_11() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  int output[10] = {123, 123, 123, 123, 123, 123, 123, 123, 123, 123};
  thrust::gather_if(thrust::seq, map, map + 10, stencil, values, output,
                    is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < 10; i++) {
    if (output[i] != values_ref[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }
  printf("test_11 run passed!\n");
}

void test_12() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  int output[10] = {123, 123, 123, 123, 123, 123, 123, 123, 123, 123};
  thrust::gather_if(map, map + 10, stencil, values, output, is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < 10; i++) {
    if (output[i] != values_ref[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }
  printf("test_12 run passed!\n");
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();
  // test_7();
  test_8();
  test_9();
  test_10();
  test_11();
  test_12();

  std::cout << "Testing char\n";
  test_A<char>();
  test_B<char>();
  test_C<char>();
  test_D<char>();

  std::cout << "Testing short int\n";
  test_A<short int>();
  test_B<short int>();
  test_C<short int>();
  test_D<short int>();

  std::cout << "Testing int\n";
  test_A<int>();
  test_B<int>();
  test_C<int>();
  test_D<int>();

  std::cout << "Testing long\n";
  test_A<long>();
  test_B<long>();
  test_C<long>();
  test_D<long>();

  std::cout << "Testing long long\n";
  test_A<long long>();
  test_B<long long>();
  test_C<long long>();
  test_D<long long>();

  std::cout << "Testing float\n";
  test_A<float>();
  test_B<float>();
  test_C<float>();
  test_D<float>();

  return 0;
}
