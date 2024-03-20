// ====------ cublas_64.cu ------------------------------- *- CUDA -* ---==== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ====------------------------------------------------------------------==== //

#include "cublas_v2.h"

void foo() {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasOperation_t transa;
  cublasOperation_t transb;
  int64_t m;
  int64_t n;
  int64_t k;
  const float *alpha_s;
  const double *alpha_d;
  const float2 *alpha_c;
  const double2 *alpha_z;
  const float *A_s;
  const double *A_d;
  const float2 *A_c;
  const double2 *A_z;
  int64_t lda;
  const float *B_s;
  const double *B_d;
  const float2 *B_c;
  const double2 *B_z;
  int64_t ldb;
  const float *beta_s;
  const double *beta_d;
  const float2 *beta_c;
  const double2 *beta_z;
  float *C_s;
  double *C_d;
  float2 *C_c;
  double2 *C_z;
  float *C1_s;
  double *C1_d;
  float2 *C1_c;
  double2 *C1_z;
  int64_t ldc;
  cublasFillMode_t uplo;
  cublasSideMode_t side;
  cublasDiagType_t diag;
  int64_t result;
  float result_s;
  double result_d;
  float2 result_c;
  double2 result_z;
  int64_t incx;
  int64_t incy;

  status = cublasIsamax_64(handle, n, A_s, lda, &result);
  status = cublasIdamax_64(handle, n, A_d, lda, &result);
  status = cublasIcamax_64(handle, n, A_c, lda, &result);
  status = cublasIzamax_64(handle, n, A_z, lda, &result);

  status = cublasIsamin_64(handle, n, A_s, lda, &result);
  status = cublasIdamin_64(handle, n, A_d, lda, &result);
  status = cublasIcamin_64(handle, n, A_c, lda, &result);
  status = cublasIzamin_64(handle, n, A_z, lda, &result);

  status = cublasSnrm2_64(handle, n, A_s, incx, &result_s);
  status = cublasDnrm2_64(handle, n, A_d, incx, &result_d);
  status = cublasScnrm2_64(handle, n, A_c, incx, &result_s);
  status = cublasDznrm2_64(handle, n, A_z, incx, &result_d);

  status = cublasSdot_64(handle, n, A_s, incx, B_s, incy, &result_s);
  status = cublasDdot_64(handle, n, A_d, incx, B_d, incy, &result_d);
  status = cublasCdotu_64(handle, n, A_c, incx, B_c, incy, &result_c);
  status = cublasCdotc_64(handle, n, A_c, incx, B_c, incy, &result_c);
  status = cublasZdotu_64(handle, n, A_z, incx, B_z, incy, &result_z);
  status = cublasZdotc_64(handle, n, A_z, incx, B_z, incy, &result_z);

  status = cublasSscal_64(handle, n, alpha_s, C_s, incx);
  status = cublasDscal_64(handle, n, alpha_d, C_d, incx);
  status = cublasCscal_64(handle, n, alpha_c, C_c, incx);
  status = cublasZscal_64(handle, n, alpha_z, C_z, incx);
  status = cublasCsscal_64(handle, n, alpha_s, C_c, incx);
  status = cublasZdscal_64(handle, n, alpha_d, C_z, incx);

  status = cublasSaxpy_64(handle, n, alpha_s, A_s, incx, C_s, incy);
  status = cublasDaxpy_64(handle, n, alpha_d, A_d, incx, C_d, incy);
  status = cublasCaxpy_64(handle, n, alpha_c, A_c, incx, C_c, incy);
  status = cublasZaxpy_64(handle, n, alpha_z, A_z, incx, C_z, incy);

  status = cublasScopy_64(handle, n, A_s, incx, C_s, incy);
  status = cublasDcopy_64(handle, n, A_d, incx, C_d, incy);
  status = cublasCcopy_64(handle, n, A_c, incx, C_c, incy);
  status = cublasZcopy_64(handle, n, A_z, incx, C_z, incy);

  status = cublasSswap_64(handle, n, C_s, incx, C1_s, incy);
  status = cublasDswap_64(handle, n, C_d, incx, C1_d, incy);
  status = cublasCswap_64(handle, n, C_c, incx, C1_c, incy);
  status = cublasZswap_64(handle, n, C_z, incx, C1_z, incy);

  status = cublasSasum_64(handle, n, A_s, incx, &result_s);
  status = cublasDasum_64(handle, n, A_d, incx, &result_d);
  status = cublasScasum_64(handle, n, A_c, incx, &result_s);
  status = cublasDzasum_64(handle, n, A_z, incx, &result_d);

  status = cublasSgemm_64(handle, transa, transb, m, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDgemm_64(handle, transa, transb, m, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCgemm_64(handle, transa, transb, m, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZgemm_64(handle, transa, transb, m, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasSsyrk_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, beta_s, C_s, ldc);
  status = cublasDsyrk_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, beta_d, C_d, ldc);
  status = cublasCsyrk_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  status = cublasZsyrk_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  status = cublasSsymm_64(handle, side, uplo, m, n, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsymm_64(handle, side, uplo, m, n, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsymm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsymm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasStrsm_64(handle, side, uplo, transa, diag, m, n, alpha_s, A_s, lda, C_s, ldc);
  status = cublasDtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_d, A_d, lda, C_d, ldc);
  status = cublasCtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_c, A_c, lda, C_c, ldc);
  status = cublasZtrsm_64(handle, side, uplo, transa, diag, m, n, alpha_z, A_z, lda, C_z, ldc);

  status = cublasChemm_64(handle, side, uplo, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZhemm_64(handle, side, uplo, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasCherk_64(handle, uplo, transa, n, k, alpha_s, A_c, lda, beta_s, C_c, ldc);
  status = cublasZherk_64(handle, uplo, transa, n, k, alpha_d, A_z, lda, beta_d, C_z, ldc);

  status = cublasSsyr2k_64(handle, uplo, transa, n, k, alpha_s, A_s, lda, B_s, ldb, beta_s, C_s, ldc);
  status = cublasDsyr2k_64(handle, uplo, transa, n, k, alpha_d, A_d, lda, B_d, ldb, beta_d, C_d, ldc);
  status = cublasCsyr2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  status = cublasZsyr2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasCher2k_64(handle, uplo, transa, n, k, alpha_c, A_c, lda, B_c, ldb, beta_s, C_c, ldc);
  status = cublasZher2k_64(handle, uplo, transa, n, k, alpha_z, A_z, lda, B_z, ldb, beta_d, C_z, ldc);

  status = cublasSgeam_64(handle, transa, transb, m, n, alpha_s, A_s, lda, beta_s, B_s, ldb, C_s, ldc);
  status = cublasDgeam_64(handle, transa, transb, m, n, alpha_d, A_d, lda, beta_d, B_d, ldb, C_d, ldc);
  status = cublasCgeam_64(handle, transa, transb, m, n, alpha_c, A_c, lda, beta_c, B_c, ldb, C_c, ldc);
  status = cublasZgeam_64(handle, transa, transb, m, n, alpha_z, A_z, lda, beta_z, B_z, ldb, C_z, ldc);

  status = cublasSdgmm_64(handle, side, m, n, A_s, lda, B_s, ldb, C_s, ldc);
  status = cublasDdgmm_64(handle, side, m, n, A_d, lda, B_d, ldb, C_d, ldc);
  status = cublasCdgmm_64(handle, side, m, n, A_c, lda, B_c, ldb, C_c, ldc);
  status = cublasZdgmm_64(handle, side, m, n, A_z, lda, B_z, ldb, C_z, ldc);
}
