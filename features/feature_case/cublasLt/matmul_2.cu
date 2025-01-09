// ===------------ matmul_2.cu --------------------------- *- CUDA -* ----=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include "cublasLt.h"

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B    + C            = alpha * A*B    + C           = D              gelu
// 2       6  10  14    5  4     -29    -7        2   14  4      -29    -7     -1      1    -0.158806 0.841194
//         7  11  15   -3 -2    2000  6000            17  6     2000  6000   2034   6012         2034     6012
//         8  12  16    1  0    3000  7000            20  8     3000  7000   3040   7016         3040     7016
//         9  13  17    p  p    4000  8000            23  10    4000  8000   4046   8020         4046     8020
// clang-format on
bool test_gelu() {
  printf("========test_gelu=========\n");
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
  float Chost[ldc * n] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

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
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_GELU;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {-0.158806, 2034, 3040, 4046, 0.841194, 6012, 7016, 8020};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);

  return !error;
}

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B    + C            = alpha * A*B    + C           = D                 + bias  =
// 2       6  10  14    5  4     -29    -7        2   14  4      -29    -7     -1      1         0.05     -0.95   1.05
//         7  11  15   -3 -2    2000  6000            17  6     2000  6000   2034   6012         200       2234   6212
//         8  12  16    1  0    3000  7000            20  8     3000  7000   3040   7016         300       3340   7316
//         9  13  17    p  p    4000  8000            23  10    4000  8000   4046   8020         400       4446   8420
// clang-format on
bool test_bias() {
  printf("========test_bias=========\n");
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
  float Chost[ldc * n] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

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

  float bias_vec_host[4] = {0.05, 200, 300, 400};
  float *bias_vec_dev;
  cudaMalloc(&bias_vec_dev, sizeof(float) * 4);
  cudaMemcpy(bias_vec_dev, bias_vec_host, sizeof(float) * 4, cudaMemcpyHostToDevice);

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_vec_dev, sizeof(bias_vec_dev));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {-0.95, 2234, 3340, 4446, 1.05, 6212, 7316, 8420};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);
  cudaFree(bias_vec_dev);

  return !error;
}

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B    + C            = alpha * A*B    + C           = D                 + bias  =                   gelu
// 2       6  10  14    5  4     -29    -7        2   14  4      -29    -7     -1      1         0.05     -0.95   1.05     -0.162640 0.895629
//         7  11  15   -3 -2    2000  6000            17  6     2000  6000   2034   6012         200       2234   6212          2234   6212
//         8  12  16    1  0    3000  7000            20  8     3000  7000   3040   7016         300       3340   7316          3340   7316
//         9  13  17    p  p    4000  8000            23  10    4000  8000   4046   8020         400       4446   8420          4446   8420
// clang-format on
bool test_gelu_bias() {
  printf("========test_gelu_bias=========\n");
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
  float Chost[ldc * n] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

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

  float bias_vec_host[4] = {0.05, 200, 300, 400};
  float *bias_vec_dev;
  cudaMalloc(&bias_vec_dev, sizeof(float) * 4);
  cudaMemcpy(bias_vec_dev, bias_vec_host, sizeof(float) * 4, cudaMemcpyHostToDevice);

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_GELU_BIAS;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_vec_dev, sizeof(bias_vec_dev));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {-0.162640, 2234, 3340, 4446, 0.895629, 6212, 7316, 8420};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);
  cudaFree(bias_vec_dev);

  return !error;
}

bool test_gelu_aux() {
  printf("========test_gelu_aux=========\n");
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
  float Chost[ldc * n] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

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

  float *aux_dev;
  const constexpr size_t aux_ld = 8;
  cudaMalloc(&aux_dev, aux_ld * n * sizeof(float));

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_GELU_AUX;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_dev, sizeof(aux_dev));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);
  float aux_host[aux_ld * n];
  cudaMemcpy(aux_host, aux_dev, aux_ld * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {-0.158806, 2034, 3040, 4046, 0.841194, 6012, 7016, 8020};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  float aux_ref[aux_ld * n] = {-1, 2034, 3040, 4046, 0, 0, 0, 0, 1, 6012, 7016, 8020, 0, 0, 0, 0};
  for (int i = 0; i < aux_ld * n; i++) {
    if ((i % aux_ld) >= m)
      continue;
    if (std::abs(aux_host[i] - aux_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");
  printf("aux:\n");
  for (int i = 0; i < aux_ld * n; i++)
    printf("%f, ", aux_host[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);
  cudaFree(aux_dev);

  return !error;
}

bool test_gelu_aux_bias() {
  printf("========test_gelu_aux_bias=========\n");
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
  float Chost[ldc * n] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

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

  float bias_vec_host[4] = {0.05, 200, 300, 400};
  float *bias_vec_dev;
  cudaMalloc(&bias_vec_dev, sizeof(float) * 4);
  cudaMemcpy(bias_vec_dev, bias_vec_host, sizeof(float) * 4, cudaMemcpyHostToDevice);

  float *aux_dev;
  const constexpr size_t aux_ld = 8;
  cudaMalloc(&aux_dev, aux_ld * n * sizeof(float));

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_vec_dev, sizeof(bias_vec_dev));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_dev, sizeof(aux_dev));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);
  float aux_host[aux_ld * n];
  cudaMemcpy(aux_host, aux_dev, aux_ld * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {-0.162640, 2234, 3340, 4446, 0.895629, 6212, 7316, 8420};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  float aux_ref[aux_ld * n] = {-0.95, 2234, 3340, 4446, 0, 0, 0, 0, 1.05, 6212, 7316, 8420, 0, 0, 0, 0};
  for (int i = 0; i < aux_ld * n; i++) {
    if ((i % aux_ld) >= m)
      continue;
    if (std::abs(aux_host[i] - aux_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");
  printf("aux:\n");
  for (int i = 0; i < aux_ld * n; i++)
    printf("%f, ", aux_host[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);
  cudaFree(aux_dev);
  cudaFree(bias_vec_dev);

  return !error;
}

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B     = alpha * A*B    = D          aux           dgelu
// 2       6  10  14    5  4        2   14  4    28  8     -0.1  -0.1     0.082964 -35.846096
//         7  11  15   -3 -2            17  6    34 12     -0.2  -0.2     27       -2
//         8  12  16    1  0            20  8    40 16      0.1   0.1     6.5      -28.200001
//         9  13  17    p  p            23  10   46 20      0.05  0.05    33       -3
// clang-format on
bool test_dgelu() {
  printf("========test_dgelu=========\n");
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  const constexpr int m = 4;
  const constexpr int n = 2;
  const constexpr int k = 3;
  const constexpr int lda = m;
  const constexpr int ldb = m;
  const constexpr int ldd = m;
  void *Adev;
  void *Bdev;
  void *Ddev;
  cudaMalloc(&Adev, lda * k * sizeof(float));
  cudaMalloc(&Bdev, ldb * n * sizeof(float));
  cudaMalloc(&Ddev, ldd * n * sizeof(float));

  float Ahost[lda * k] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  float Bhost[ldb * n] = {5, -3, 1, 99, 4, -2, 0, 99};

  cudaMemcpy(Adev, Ahost, lda * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, ldb * n * sizeof(float), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL,
                         Bdesc_col_major = NULL,
                         Ddesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, k, n, ldb);
  cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, m, n, ldd);

  float alpha = 2;
  float beta = 0;

  float *aux_dev;
  size_t aux_ld = 4;
  cudaMalloc(&aux_dev, aux_ld * n * sizeof(float));
  float aux_host[8] = {-0.1, -0.2, 0.1, 0.05, -0.1, -0.2, 0.1, 0.05};
  cudaMemcpy(aux_dev, aux_host, aux_ld * n * sizeof(float), cudaMemcpyHostToDevice);

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_DGELU;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_dev, sizeof(aux_dev));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Ddev, Ddesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {11.773392, 11.646420, 23.180870, 24.833599, 3.363826, 4.110501, 9.272348, 10.797216};
  for (int i = 0; i < ldd * n; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");

  if (error) {
    printf("error\n");
  } else {
    printf("success\n");
  }

  cublasLtDestroy(ltHandle);
  cublasLtMatrixLayoutDestroy(Adesc_col_major);
  cublasLtMatrixLayoutDestroy(Bdesc_col_major);
  cublasLtMatrixLayoutDestroy(Ddesc_col_major);
  cudaFree(Adev);
  cudaFree(Bdev);
  cudaFree(Ddev);
  cudaFree(aux_dev);

  return !error;
}

// clang-format off
// A (4*3)     B (3*2)
// 6 10 14     5  4
// 7 11 15    -3 -2
// 8 12 16     1  0
// 9 13 17     p  p
//
// alpha * A          * B    + C            = alpha * A*B    + C           = D              gelu
// 2       6  10  14    5  4     -29    -7        2   14  4      -29    -7     -1      1    -0.158806 0.841194
//         7  11  15   -3 -2    2000  6000            17  6     2000  6000   2034   6012         2034     6012
//         8  12  16    1  0    3000  7000            20  8     3000  7000   3040   7016         3040     7016
//         9  13  17    p  p    4000  8000            23  10    4000  8000   4046   8020         4046     8020
// clang-format on
bool test_batch() {
  printf("========test_batch=========\n");
  const constexpr int batch = 2;
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
  cudaMalloc(&Adev, lda * k * sizeof(float) * batch);
  cudaMalloc(&Bdev, ldb * n * sizeof(float) * batch);
  cudaMalloc(&Cdev, ldc * n * sizeof(float) * batch);
  cudaMalloc(&Ddev, ldd * n * sizeof(float) * batch);

  float Ahost[lda * k * batch] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  float Bhost[ldb * n * batch] = {5, -3, 1, 99, 4, -2, 0, 99, 5, -3, 1, 99, 4, -2, 0, 99};
  float Chost[ldc * n * batch] = {-29, 2000, 3000, 4000, -7, 6000, 7000, 8000, -29, 2000, 3000, 4000, -7, 6000, 7000, 8000};

  cudaMemcpy(Adev, Ahost, lda * k * sizeof(float) * batch, cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, ldb * n * sizeof(float) * batch, cudaMemcpyHostToDevice);
  cudaMemcpy(Cdev, Chost, ldc * n * sizeof(float) * batch, cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL,
                         Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL,
                         Ddesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, m, k, lda);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, k, n, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32F, m, n, ldc);
  cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, m, n, ldd);

  cublasLtMatrixLayoutSetAttribute(Adesc_col_major, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col_major, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col_major, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
  cublasLtMatrixLayoutSetAttribute(Ddesc_col_major, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
  int64_t offset_a = 12;
  int64_t offset_b = 8;
  int64_t offset_c = 8;
  int64_t offset_d = 8;
  cublasLtMatrixLayoutSetAttribute(Adesc_col_major, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &offset_a, sizeof(offset_a));
  cublasLtMatrixLayoutSetAttribute(Bdesc_col_major, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &offset_b, sizeof(offset_b));
  cublasLtMatrixLayoutSetAttribute(Cdesc_col_major, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &offset_c, sizeof(offset_c));
  cublasLtMatrixLayoutSetAttribute(Ddesc_col_major, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &offset_d, sizeof(offset_d));

  float alpha = 2;
  float beta = 1;

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_GELU;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n * batch];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float) * batch, cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n * batch] = {-0.158806, 2034, 3040, 4046, 0.841194, 6012, 7016, 8020, -0.158806, 2034, 3040, 4046, 0.841194, 6012, 7016, 8020};
  for (int i = 0; i < ldd * n * batch; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < ldd * n * batch; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);

  return !error;
}

// clang-format off
// alpha * A    * B           + C                   = alpha * A*B        + C                   = D
// 2       6 10    5  7 -1 3   1000 3000 5000 7000    2       0 22 14 28   1000 3000 5000 7000   1000 3044 5028 7056
//         7 11   -3 -2  2 1   2000 4000 6000 8000            2 27 15 32   2000 4000 6000 8000   2004 4054 6030 8064
// clang-format on
bool test_bgradb() {
  printf("========test_bgradb=========\n");
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  void *Adev;
  void *Bdev;
  void *Cdev;
  void *Ddev;
  cudaMalloc(&Adev, 4 * sizeof(float));
  cudaMalloc(&Bdev, 8 * sizeof(float));
  cudaMalloc(&Cdev, 8 * sizeof(float));
  cudaMalloc(&Ddev, 8 * sizeof(float));

  float Ahost[4] = {6, 7, 10, 11};
  float Bhost[8] = {5, 7, -1, 3, -3, -2,  2,  1};
  float Chost[8] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};

  cudaMemcpy(Adev, Ahost, 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bdev, Bhost, 8 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Cdev, Chost, 8 * sizeof(float), cudaMemcpyHostToDevice);

  cublasLtMatrixLayout_t Adesc_col_major = NULL,
                         Bdesc_col_major = NULL,
                         Cdesc_col_major = NULL,
                         Ddesc_col_major = NULL;
  cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, 2, 2, 2);
  cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, 4, 2, 4);
  cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32F, 2, 4, 2);
  cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, 2, 4, 2);

  float alpha = 2;
  float beta = 1;
  float *bias_vec_dev;
  cudaMalloc(&bias_vec_dev, sizeof(float) * 4);

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  auto transb = CUBLAS_OP_T;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BGRADB;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_vec_dev, sizeof(bias_vec_dev));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[8];
  cudaMemcpy(Dhost, Ddev, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  float bias_vec_host[4];
  cudaMemcpy(bias_vec_host, bias_vec_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[8] = {1000, 2004, 3044, 4054, 5028, 6030, 7056, 8064};
  for (int i = 0; i < 8; i++) {
    if (std::abs(Dhost[i] - D_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  float bias_vec_ref[4] = {2, 5, 1, 4};
  for (int i = 0; i < 2; i++) {
    if (std::abs(bias_vec_host[i] - bias_vec_ref[i]) > 0.01) {
      error = true;
      break;
    }
  }

  printf("d:\n");
  for (int i = 0; i < 8; i++)
    printf("%f, ", Dhost[i]);
  printf("\n");
  printf("bias_vec:\n");
  for (int i = 0; i < 4; i++)
    printf("%f, ", bias_vec_host[i]);
  printf("\n");

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
  cudaFree(Cdev);
  cudaFree(Ddev);
  cudaFree(bias_vec_dev);
  
  return !error;
}

int main() {
  bool pass = true;
  pass = test_gelu() && pass;
  pass = test_bias() && pass;
  pass = test_gelu_bias() && pass;
  pass = test_gelu_aux() && pass;
  pass = test_gelu_aux_bias() && pass;
  pass = test_dgelu() && pass;
#ifndef DPCT_USM_LEVEL_NONE
  pass = test_batch() && pass;
#endif
  pass = test_bgradb() && pass;

  if (pass)
    printf("matmul_2 all passed.\n");

  return pass ? 0 : 1;
}
