// ====------ thrust_scatter_if.cu------------- *- CUDA -*-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------- ---------------===//

#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

struct is_even {
  __host__ __device__ bool operator()(int x) const { return (x % 2) == 0; }
};

void test_1() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  is_even pred;
  thrust::scatter_if(thrust::host, V, V + 8, M, S, D, pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (D[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_1 run pass!\n");
}

void test_2() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  is_even pred;
  thrust::scatter_if(V, V + 8, M, S, D, pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (D[i] != ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run pass!\n");
}

void test_3() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  thrust::device_vector<int> d_V(V, V + N);
  thrust::device_vector<int> d_M(M, M + N);
  thrust::device_vector<int> d_S(S, S + N);
  thrust::device_vector<int> d_D(N);

  is_even pred;
  thrust::scatter_if(thrust::device, d_V.begin(), d_V.end(), d_M.begin(),
                     d_S.begin(), d_D.begin(), pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run pass!\n");
}

void test_4() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  thrust::device_vector<int> d_V(V, V + N);
  thrust::device_vector<int> d_M(M, M + N);
  thrust::device_vector<int> d_S(S, S + N);
  thrust::device_vector<int> d_D(N);

  is_even pred;
  thrust::scatter_if(d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(),
                     d_D.begin(), pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run pass!\n");
}

void test_5() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  thrust::host_vector<int> h_V(V, V + N);
  thrust::host_vector<int> h_M(M, M + N);
  thrust::host_vector<int> h_S(S, S + N);
  thrust::host_vector<int> h_D(N);

  is_even pred;
  thrust::scatter_if(thrust::host, h_V.begin(), h_V.end(), h_M.begin(),
                     h_S.begin(), h_D.begin(), pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (h_D[i] != ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }
  printf("test_5 run pass!\n");
}

void test_6() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {2, 1, 2, 1, 2, 1, 2, 1};
  thrust::host_vector<int> h_V(V, V + N);
  thrust::host_vector<int> h_M(M, M + N);
  thrust::host_vector<int> h_S(S, S + N);
  thrust::host_vector<int> h_D(N);

  is_even pred;
  thrust::scatter_if(h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(),
                     h_D.begin(), pred);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (h_D[i] != ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  printf("test_6 run pass!\n");
}

void test_7() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::scatter_if(thrust::host, V, V + 8, M, S, D);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (D[i] != ref[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }
  printf("test_7 run pass!\n");
}

void test_8() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  is_even pred;
  thrust::scatter_if(V, V + 8, M, S, D);

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (D[i] != ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }
  printf("test_8 run pass!\n");
}

void test_9() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::device_vector<int> d_V(V, V + N);
  thrust::device_vector<int> d_M(M, M + N);
  thrust::device_vector<int> d_S(S, S + N);
  thrust::device_vector<int> d_D(D, D + N);

  thrust::scatter_if(thrust::device, d_V.begin(), d_V.end(), d_M.begin(),
                     d_S.begin(), d_D.begin());

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }
  printf("test_9 run pass!\n");
}

void test_10() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::device_vector<int> d_V(V, V + N);
  thrust::device_vector<int> d_M(M, M + N);
  thrust::device_vector<int> d_S(S, S + N);
  thrust::device_vector<int> d_D(D, D + N);

  thrust::scatter_if(d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(),
                     d_D.begin());

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }
  printf("test_10 run pass!\n");
}

void test_11() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::host_vector<int> d_V(V, V + N);
  thrust::host_vector<int> d_M(M, M + N);
  thrust::host_vector<int> d_S(S, S + N);
  thrust::host_vector<int> d_D(D, D + N);

  thrust::scatter_if(thrust::host, d_V.begin(), d_V.end(), d_M.begin(),
                     d_S.begin(), d_D.begin());

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }
  printf("test_11 run pass!\n");
}

void test_12() {
  const int N = 8;
  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::host_vector<int> d_V(V, V + N);
  thrust::host_vector<int> d_M(M, M + N);
  thrust::host_vector<int> d_S(S, S + N);
  thrust::host_vector<int> d_D(D, D + N);

  thrust::scatter_if(d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(),
                     d_D.begin());

  int ref[N] = {10, 30, 50, 70, 0, 0, 0, 0};

  for (int i = 0; i < N; i++) {
    if (d_D[i] != ref[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }
  printf("test_12 run pass!\n");
}

int main() {
  test_1();
  test_2();

  test_3();
  test_4();

  test_5();
  test_6();

  test_7();
  test_8();

  test_9();
  test_10();

  test_11();
  test_12();

  return 0;
}
