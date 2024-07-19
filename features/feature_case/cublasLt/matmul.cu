// ===------------ matmul.cu ----------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <cublasLt.h>
#include <cstdint>
#include <stdexcept>

const constexpr int COL_TURING = 0;
const constexpr int COL_AMPERE = 1;

// The original source of below two functions was under the license below:
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Repo: https://github.com/TimDettmers/bitsandbytes.git
inline int checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        //throw std::logic_error("cuBLAS API failed");
        return 1;
    }
    return 0;
}

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
{
    int has_error = 0;
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t col_ampere = CUBLASLT_ORDER_COL32_2R_4R4;

    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));

    has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));
    else
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_ampere, sizeof(col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, NULL, NULL, 0, 0));
    }
    else
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
      else
      {
        has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &alphaVec, sizeof(alphaVec)));
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
    }

    cudaStreamSynchronize(0);

    if (Cdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) has_error |= checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if(has_error == 1)
      printf("error detected");

    return has_error;
}

void transform(cublasLtHandle_t ltHandle, const void *in, int ld_in,
               cublasLtMatrixLayout_t layout_in, void *out, int ld_out,
               cublasLtMatrixLayout_t layout_out) {
  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F);
  float alpha = 1.0f, beta = 0.0f;
  cublasLtMatrixTransform(ltHandle, transform_desc, &alpha, in, layout_in,
                          &beta, NULL, NULL, out, layout_out, 0);
  cublasLtMatrixTransformDescDestroy(transform_desc);
}

// igemmlt<COL_TURING, 32, 0>
bool test1() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int32_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int32_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col4_4r2_8c, ((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int32_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col4_4r2_8c, CUDA_R_8I, k, n,
                             ((n + 8 - 1) / 8) * 8 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_32I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col4_4r2_8c = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col4_4r2_8c,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col4_4r2_8c,
                                   sizeof(col4_4r2_8c));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  // Matmul
  igemmlt<COL_TURING, 32, 0>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                             nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                             m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int32_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int32_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col4_4r2_8c);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);

  return !error;
}

// igemmlt<COL_TURING, 8, 0>
bool test2() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int8_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_8I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int8_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col4_4r2_8c, ((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int8_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col4_4r2_8c, CUDA_R_8I, k, n,
                             ((n + 8 - 1) / 8) * 8 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_8I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col4_4r2_8c = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col4_4r2_8c,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col4_4r2_8c,
                                   sizeof(col4_4r2_8c));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  // Matmul
  igemmlt<COL_TURING, 8, 0>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                            nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                            m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int8_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int8_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int8_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col4_4r2_8c);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);

  return !error;
}

// igemmlt<COL_TURING, 8, 1>
bool test3() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int8_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_8I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col4_4r2_8c = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col4_4r2_8c;
  int8_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col4_4r2_8c, ((n + 8 - 1) / 8) * 8 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int8_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col4_4r2_8c, CUDA_R_8I, k, n,
                             ((n + 8 - 1) / 8) * 8 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_8I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col4_4r2_8c = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col4_4r2_8c,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col4_4r2_8c,
                                   sizeof(col4_4r2_8c));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col4_4r2_8c, 8 * 32,
            Bdesc_col4_4r2_8c);

  float *alpha;
  cudaMallocManaged(&alpha, 4 * sizeof(float));
  alpha[0] = 0;
  alpha[1] = 1;
  alpha[2] = 2;
  alpha[3] = 3;

  // Matmul
  igemmlt<COL_TURING, 8, 1>(ltHandle, m, n, k, A_col32, B_col4_4r2_8c, C_col32,
                            alpha, m * 32, ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int8_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int8_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int8_t C_ref[m * n] = {0, 17, 40, 69, 0, 6, 16, 30};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col4_4r2_8c);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);
  cudaFree(alpha);

  return !error;
}

// igemmlt<COL_AMPERE, 32, 0>
bool test4() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int32_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int32_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col32_2r_4r4,
             ((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int32_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col32_2r_4r4, CUDA_R_8I, k, n,
                             ((n + 32 - 1) / 32) * 32 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_32I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col32_2r_4r4 = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col32_2r_4r4,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col32_2r_4r4,
                                   sizeof(col32_2r_4r4));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  // Matmul
  igemmlt<COL_AMPERE, 32, 0>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4,
                             C_col32, nullptr, m * 32,
                             ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int32_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int32_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col32_2r_4r4);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);

  return !error;
}

// igemmlt<COL_AMPERE, 8, 0>
bool test5() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int8_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_8I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int8_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col32_2r_4r4,
             ((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int8_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col32_2r_4r4, CUDA_R_8I, k, n,
                             ((n + 32 - 1) / 32) * 32 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_8I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col32_2r_4r4 = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col32_2r_4r4,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col32_2r_4r4,
                                   sizeof(col32_2r_4r4));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  // Matmul
  igemmlt<COL_AMPERE, 8, 0>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4, C_col32,
                            nullptr, m * 32, ((n + 8 - 1) / 8) * 8 * 32,
                            m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int8_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int8_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int8_t C_ref[m * n] = {14, 17, 20, 23, 4, 6, 8, 10};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col32_2r_4r4);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);

  return !error;
}

// igemmlt<COL_AMPERE, 8, 1>
bool test6() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  int lda = m;
  int ldb = n;
  int ldc = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  cudaMalloc(&Adev, m * k * sizeof(int8_t));
  cudaMalloc(&Bdev, n * k * sizeof(int8_t));
  cudaMalloc(&Cdev, m * n * sizeof(int8_t));

  int8_t Ahost[m * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int8_t Bhost[n * k] = {5, 4, -3, -2, 1, 0};

  cudaMemcpy(Adev, Ahost, m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, n * k * sizeof(int8_t), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL, Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_8I, m, n, ldc);

  // Convert A and B
  cublasLtMatrixLayout_t Adesc_col32 = NULL, Bdesc_col32_2r_4r4 = NULL,
                         Cdesc_col32 = NULL;
  int8_t *A_col32, *B_col32_2r_4r4;
  int8_t *C_col32;
  cudaMalloc(&A_col32, m * 32 * sizeof(std::int8_t));
  cudaMalloc(&B_col32_2r_4r4,
             ((n + 32 - 1) / 32) * 32 * 32 * sizeof(std::int8_t));
  cudaMalloc(&C_col32, m * 32 * sizeof(std::int8_t));
  cublasLtMatrixLayoutCreate(&Adesc_col32, CUDA_R_8I, m, k, m * 32);
  cublasLtMatrixLayoutCreate(&Bdesc_col32_2r_4r4, CUDA_R_8I, k, n,
                             ((n + 32 - 1) / 32) * 32 * 32);
  cublasLtMatrixLayoutCreate(&Cdesc_col32, CUDA_R_8I, m, n, m * 32);
  cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t col32_2r_4r4 = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtMatrixLayoutSetAttribute(Adesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col32_2r_4r4,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &col32_2r_4r4,
                                   sizeof(col32_2r_4r4));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col32, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &col32, sizeof(col32));

  transform(ltHandle, Adev, lda, Adesc_col_major, A_col32, m * 32, Adesc_col32);
  transform(ltHandle, Bdev, ldb, Bdesc_col_major, B_col32_2r_4r4, 8 * 32,
            Bdesc_col32_2r_4r4);

  float *alpha;
  cudaMallocManaged(&alpha, 4 * sizeof(float));
  alpha[0] = 0;
  alpha[1] = 1;
  alpha[2] = 2;
  alpha[3] = 3;

  // Matmul
  igemmlt<COL_AMPERE, 8, 1>(ltHandle, m, n, k, A_col32, B_col32_2r_4r4, C_col32,
                            alpha, m * 32, ((n + 8 - 1) / 8) * 8 * 32, m * 32);

  // Convert C
  transform(ltHandle, C_col32, m * 32, Cdesc_col32, Cdev, ldc, Cdesc_col_major);
  cudaStreamSynchronize(0);

  // Check result
  int8_t Chost[m * n];
  cudaMemcpy(Chost, Cdev, m * n * sizeof(int8_t), cudaMemcpyDeviceToHost);

  bool error = false;
  int8_t C_ref[m * n] = {0, 17, 40, 69, 0, 6, 16, 30};
  for (int i = 0; i < m * n; i++) {
    if (Chost[i] != C_ref[i]) {
      error = true;
      break;
    }
  }
  printf("c:\n");
  for (int i = 0; i < m * n; i++)
    printf("%d, ", Chost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col32);
  cublasLtMatrixLayoutDestroy(Bdesc_col32_2r_4r4);
  cublasLtMatrixLayoutDestroy(Cdesc_col32);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Cdev);
  cudaFree(alpha);

  return !error;
}

void fgemmlt(cublasLtHandle_t ltHandle, int m, int n, int k,
             const float *A, const float *B, const float *C, float *D,
             float *alpha, float *beta,
             int lda, int ldb, int ldc, int ldd,
             cublasLtMatrixLayout_t Adesc,
             cublasLtMatrixLayout_t Bdesc,
             cublasLtMatrixLayout_t Cdesc,
             cublasLtMatrixLayout_t Ddesc,
             float *amax_d) {
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  float *scale_a;
  float *scale_b;
  float *scale_d;
  cudaMallocManaged(&scale_a, sizeof(float));
  cudaMallocManaged(&scale_b, sizeof(float));
  cudaMallocManaged(&scale_d, sizeof(float));
  scale_a[0] = 3;
  scale_b[0] = 5;
  scale_d[0] = 7;

  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scale_d, sizeof(scale_d));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d));

  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_RELU;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));

  cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, NULL, NULL, 0, 0);

  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);
}

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B    + C            = alpha * A*B    + C           = D
// 2*3*5   6  10  14    5  4  -10000 -5000       30   14  4   -10000 -5000  -9580  -4880
//         7  11  15   -3 -2    2000  6000            17  6     2000  6000   2510   6180
//         8  12  16    1  0    3000  7000            20  8     3000  7000   3600   7240
//         9  13  17    p  p    4000  8000            23  10    4000  8000   4690   8300
// scale_d *  D           =  D
//       7 * -9580 -4880    -67060  -34160
//            2510  6180     17570   43260
//            3600  7240     25200   50680
//            4690  8300     32830   58100
// clang-format on

bool test7() {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  const constexpr int lda = m;
  const constexpr int ldb = m;
  const constexpr int ldc = m;
  const constexpr int ldd = m;
  void *Adev;
  void *Bdev;
  void *Cdev;
  void *Ddev;
  cudaMalloc(&Adev, lda * k * sizeof(float));
  cudaMalloc(&Bdev, ldb * n * sizeof(float));
  cudaMalloc(&Cdev, ldc * n * sizeof(float));
  cudaMalloc(&Ddev, ldd * n * sizeof(float));

  float Ahost[lda * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  float Bhost[ldb * n] = {5, -3, 1, 99, 4, -2, 0, 99};
  float Chost[ldc * n] = {-1000, 2000, 3000, 4000, -5000, 6000, 7000, 8000};

  cudaMemcpy(Adev, Ahost, lda * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, ldb * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Cdev, Chost, ldc * n * sizeof(float), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL,
                         Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL,
                         Ddesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, k, n, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32F, m, n, ldc);
  cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, m, n, ldd);

  float alpha = 2;
  float beta = 1;

  // Matmul

  float *amax_d;
  cudaMallocManaged(&amax_d, sizeof(float));

  fgemmlt(ltHandle, m, n, k, (const float *)Adev, (const float *)Bdev, (const float *)Cdev, (float *)Ddev,
          &alpha, &beta, lda, ldb, ldc, ldd, Adesc_col_major, Bdesc_col_major, Cdesc_col_major, Ddesc_col_major, amax_d);
  cudaStreamSynchronize(0);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {0, 17570, 25200, 32830, 0, 43260, 50680, 58100};
  for (int i = 0; i < ldd * n; i++) {
    if (Dhost[i] != D_ref[i]) {
      error = true;
      break;
    }
  }
  if (*amax_d != 8300)
    error = true;

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");
  printf("amax_d:%f\n", *amax_d);

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Cdesc_col_major);
  cublasLtMatrixLayoutDestroy(Ddesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Ddev);
  cudaFree(amax_d);

  return !error;
}


// clang-format off
// A (4*3)    B (2*3)
// 6 10 14    5 -3 1
// 7 11 15    4 -2 0
// 8 12 16
// 9 13 17
//
// alpha * A          * op(B)   = alpha * C       =  C
// 0       6  10  14    5  4      0       14  4      0   0
// 1       7  11  15   -3 -2      1       17  6      17  6
// 2       8  12  16    1  0      2       20  8      40  16
// 3       9  13  17              3       23  10     69  30
//
// alpha * A          * op(B)   = alpha * C       =  C
// 1       6  10  14    5  4      1       14  4      14  4
//         7  11  15   -3 -2              17  6      17  6
//         8  12  16    1  0              20  8      20  8
//         9  13  17                      23  10     23  10
// clang-format on

int main() {
  bool pass = true;
  pass = test1() && pass;
  pass = test2() && pass;
  pass = test3() && pass;
  pass = test4() && pass;
  pass = test5() && pass;
  pass = test6() && pass;
  pass = test7() && pass;
  return pass ? 0 : 1;
}
