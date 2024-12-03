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
  cudaFree(Ddev);

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
// alpha * A          * B    + C           = alpha * A*B    + C           = D              gelu
// 2       6  10  14    5  4     -29   -7       2   14  4       -29   -7      -1   1    -0.158806 0.841194
//         7  11  15   -3 -2   -33.5  -13           17  6     -33.5  -13     0.5  -1         2034     6012
//         8  12  16    1  0   -41.1  -14           20  8     -41.1  -14    -1.1   2         3040     7016
//         9  13  17    p  p   -44.2  -23           23  10    -44.2  -23     1.8  -3         4046     8020
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
  float Chost[ldc * n] = {-29, -7, -33.5, -13, -41.1, -14, -44.2, -23};

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
  size_t aux_ld = 4;
  cudaMalloc(&aux_dev, aux_ld * n * sizeof(float));
  float aux_host[8] = {-1, 2034, 3040, 4046, 1, 6012, 7016, 8020};
  cudaMemcpy(aux_dev, aux_host, aux_ld * n * sizeof(float), cudaMemcpyHostToDevice);

  // Matmul
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_DGELU;
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_dev, sizeof(aux_dev));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major, &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);
  cudaStreamSynchronize(0);
  cublasLtMatmulDescDestroy(matmulDesc);

  // Check result
  float Dhost[ldd * n];
  cudaMemcpy(Dhost, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

  bool error = false;
  float D_ref[ldd * n] = {0.082964, 27.000000, 6.500000, 33.000000, -35.846096, -2.000000, -28.200001, -3.000000};
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
  cudaFree(Ddev);
  cudaFree(bias_vec_dev);

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
  cudaFree(Ddev);
  cudaFree(bias_vec_dev);

  return !error;
}

int main() {
  bool pass = true;
  pass = test_gelu() && pass;
  pass = test_gelu_aux() && pass;
  pass = test_dgelu() && pass;
  pass = test_gelu_bias() && pass;
  pass = test_gelu_aux_bias() && pass;
  pass = test_bias() && pass;
  return pass ? 0 : 1;
}
