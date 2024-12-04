#include "cublaslt.h"

bool test_bgradb() {
    printf("========test_bgradb=========\n");
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    const constexpr int m = 4;
    const constexpr int n = 2;
    const constexpr int k = 3;
    const constexpr int lda = m;
    const constexpr int ldb = m;
    const constexpr int ldc = m;
    const constexpr int ldd = m;

    void *Adev, *Bdev, *Cdev, *Ddev;
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

    cublasLtMatrixLayout_t Adesc_col_major, Bdesc_col_major, Cdesc_col_major, Ddesc_col_major;
    cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32F, m, n, ldc);
    cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, m, n, ldd);

    float alpha = 1.0;
    float beta = 1.0;

    // Test `CUBLASLT_EPILOGUE_BGRADB`
    printf("Testing CUBLASLT_EPILOGUE_BGRADB\n");
    cublasLtMatmulDesc_t matmulDesc_bgradb;
    cublasLtMatmulDescCreate(&matmulDesc_bgradb, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtEpilogue_t ep_bgradb = CUBLASLT_EPILOGUE_BGRADB;
    cublasLtMatmulDescSetAttribute(matmulDesc_bgradb, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep_bgradb, sizeof(ep_bgradb));

    cublasLtMatmul(ltHandle, matmulDesc_bgradb, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major,
                   &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);

    cudaStreamSynchronize(0);
    cublasLtMatmulDescDestroy(matmulDesc_bgradb);

    float Dhost_bgradb[ldd * n];
    cudaMemcpy(Dhost_bgradb, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);

    float D_ref_bgradb[ldd * n] = {-1, 1, 2034, 6012, 3040, 7016, 4046, 8020};
    bool error_bgradb = false;
    for (int i = 0; i < ldd * n; i++) {
        if (std::abs(Dhost_bgradb[i] - D_ref_bgradb[i]) > 0.01) {
            error_bgradb = true;
            break;
        }
    }

    printf("D (BGRADB):\n");
    for (int i = 0; i < ldd * n; i++) {
        printf("%f, ", Dhost_bgradb[i]);
    }
    printf("\n");
    if (error_bgradb) {
        printf("Error in BGRADB result\n");
    } else {
        printf("BGRADB test passed\n");
    }


    //Cleanup
    cublasLtDestroy(ltHandle);
    cublasLtMatrixLayoutDestroy(Adesc_col_major);
    cublasLtMatrixLayoutDestroy(Bdesc_col_major);
    cublasLtMatrixLayoutDestroy(Cdesc_col_major);
    cublasLtMatrixLayoutDestroy(Ddesc_col_major);
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
    cudaFree(Ddev);
    
    return !error_bgradb;
}


bool test_dgelu(){
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

    void *Adev, *Bdev, *Cdev, *Ddev;
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

    cublasLtMatrixLayout_t Adesc_col_major, Bdesc_col_major, Cdesc_col_major, Ddesc_col_major;
    cublasLtMatrixLayoutCreate(&Adesc_col_major, CUDA_R_32F, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc_col_major, CUDA_R_32F, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc_col_major, CUDA_R_32F, m, n, ldc);
    cublasLtMatrixLayoutCreate(&Ddesc_col_major, CUDA_R_32F, m, n, ldd);

    float alpha = 1.0;
    float beta = 1.0;

    // Test `CUBLASLT_EPILOGUE_DGELU`
    printf("Testing CUBLASLT_EPILOGUE_DGELU\n");
    cublasLtMatmulDesc_t matmulDesc_dgelu;
    cublasLtMatmulDescCreate(&matmulDesc_dgelu, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtEpilogue_t ep_dgelu = CUBLASLT_EPILOGUE_DGELU;
    cublasLtMatmulDescSetAttribute(matmulDesc_dgelu, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep_dgelu, sizeof(ep_dgelu));

    cublasLtMatmul(ltHandle, matmulDesc_dgelu, &alpha, Adev, Adesc_col_major, Bdev, Bdesc_col_major,
                   &beta, Cdev, Cdesc_col_major, Ddev, Ddesc_col_major, NULL, NULL, 0, 0);

    cudaStreamSynchronize(0);
    cublasLtMatmulDescDestroy(matmulDesc_dgelu);

    float Dhost_dgelu[ldd * n];
    cudaMemcpy(Dhost_dgelu, Ddev, ldd * n * sizeof(float), cudaMemcpyDeviceToHost);
    //dGELU(x)=0.5·(1+erf(x/root(2)) + x/root(2pi)e^(-x.sq)?
    float D_ref_dgelu[ldd * n] = {-0.158806, 0.841194, 1, 1, 1, 1, 1, 1};
    bool error_dgelu = false;
    for (int i = 0; i < ldd * n; i++) {
        if (std::abs(Dhost_dgelu[i] - D_ref_dgelu[i]) > 0.01) {
            error_dgelu = true;
            break;
        }
    }

    printf("D (DGELU):\n");
    for (int i = 0; i < ldd * n; i++) {
        printf("%f, ", Dhost_dgelu[i]);
    }
    printf("\n");
    if (error_dgelu) {
        printf("Error in DGELU result\n");
    } else {
        printf("DGELU test passed\n");
    }

    // Cleanup
    cublasLtDestroy(ltHandle);
    cublasLtMatrixLayoutDestroy(Adesc_col_major);
    cublasLtMatrixLayoutDestroy(Bdesc_col_major);
    cublasLtMatrixLayoutDestroy(Cdesc_col_major);
    cublasLtMatrixLayoutDestroy(Ddesc_col_major);
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
    cudaFree(Ddev);

    return !error_dgelu;
}

int main() {
  bool pass = true;
  pass = test_dgelu() && pass;
  pass = test_bgradb() && pass;
  return pass ? 0 : 1;
}
