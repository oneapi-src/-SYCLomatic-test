// ===------- asm_membar_fence.cu ------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <stdio.h>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__global__ void producer_kernel_membar_gl(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("membar.gl;" : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_membar_gl(int *data, int *flag, int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("membar.gl;" : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_membar_cta(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("membar.cta;" : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_membar_cta(int *data, int *flag, int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("membar.cta;" : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_membar_sys(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("membar.sys;" : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_membar_sys(int *data, int *flag, int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("membar.sys;" : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_sc_cta(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.sc.cta; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_sc_cta(int *data, int *flag,
                                             int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.sc.cta; " : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_sc_gpu(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.sc.gpu; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_sc_gpu(int *data, int *flag,
                                             int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.sc.gpu; " : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_sc_sys(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.sc.sys; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_sc_sys(int *data, int *flag,
                                             int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.sc.sys; " : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_acq_rel_cta(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.acq_rel.cta; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_acq_rel_cta(int *data, int *flag,
                                                  int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.acq_rel.cta; " : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_acq_rel_gpu(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.acq_rel.cta; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_acq_rel_gpu(int *data, int *flag,
                                                  int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.acq_rel.gpu; " : : : "memory");
    *output = *data;
  }
}

__global__ void producer_kernel_fence_acq_rel_sys(int *data, int *flag) {
  if (threadIdx.x == 0) {
    *data = 0xABCD;

    asm volatile("fence.acq_rel.sys; " : : : "memory");
    *flag = 1;
  }
}

__global__ void consumer_kernel_fence_acq_rel_sys(int *data, int *flag,
                                                  int *output) {
  if (threadIdx.x == 0) {
    while (atomicAdd(flag, 0) == 0) {
    }

    asm volatile("fence.acq_rel.sys; " : : : "memory");
    *output = *data;
  }
}

bool kernel_membar_gl_test() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_membar_gl<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_membar_gl<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_membar_sys_test() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_membar_sys<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_membar_sys<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_membar_cta_test() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_membar_cta<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_membar_cta<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_kernel_fence_sc_cta() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_sc_cta<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_sc_cta<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_kernel_fence_sc_device() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_sc_gpu<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_sc_gpu<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_kernel_fence_sc_sys() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_sc_sys<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_sc_sys<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_fence_acq_rel_cta() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_acq_rel_cta<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_acq_rel_cta<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_fence_acq_rel_device() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_acq_rel_cta<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_acq_rel_cta<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

bool kernel_fence_acq_rel_sys() {
  int *d_data, *d_flag, *d_output;
  int h_data = 0, h_flag = 0, h_output = 0;

  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

  producer_kernel_fence_acq_rel_sys<<<1, 32>>>(d_data, d_flag);
  consumer_kernel_fence_acq_rel_sys<<<1, 32>>>(d_data, d_flag, d_output);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_flag);
  cudaFree(d_output);

  if (h_output == 0xABCD) {
    return true;
  } else {
    return false;
  }
}

int main() {
  TEST(kernel_membar_gl_test);
  TEST(kernel_membar_cta_test);
  TEST(kernel_membar_sys_test);

  TEST(kernel_kernel_fence_sc_cta);
  TEST(kernel_kernel_fence_sc_device);
  TEST(kernel_kernel_fence_sc_sys);

  TEST(kernel_fence_acq_rel_cta);
  TEST(kernel_fence_acq_rel_device);
  TEST(kernel_fence_acq_rel_sys);

  return 0;
}
