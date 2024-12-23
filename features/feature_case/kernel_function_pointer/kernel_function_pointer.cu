// ====------ kernel_function_pointer.cu--------------- *- CUDA -*---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


template<typename T>
__global__ void vectorTemplateAdd(const T *A, T *B, T *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
using fpt = void(*)(const T *, T*, T*, int);

void foo() {
    int N = 10;
    size_t size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    fpt<int> fp = vectorAdd;

    void *kernel_func = (void *)&vectorAdd;

    fp<<<1, 10>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    void *args[4];
    args[0] = &d_A;
    args[1] = &d_B;
    args[2] = &d_C;
    args[3] = &N;

    cudaLaunchKernel((void *)fp, 1, 10, args, 0, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }


    cudaLaunchKernel<void(const int*, int*, int*, int)>(fp, 1, 10, args, 0, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

template<typename T>
void goo(fpt<T> p) {
    int N = 10;
    size_t size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    p<<<1, 10>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}


template <typename T>
void hoo() {
  fpt<int> a = vectorTemplateAdd;
  goo<T>(vectorTemplateAdd);
}

int main() {
  hoo<int>();
  foo();
  std::cout << "test success" << std::endl;
  return 0;
}
