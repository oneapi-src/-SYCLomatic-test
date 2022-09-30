// ====------ cusolverDnLn-part2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>


int main(int argc, char *argv[])
{
    cusolverDnHandle_t* cusolverH = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = CUSOLVER_STATUS_NOT_INITIALIZED;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    int m = 0;
    int n = 0;
    int k = 0;
    int nrhs = 0;
    float A_f = 0;
    double A_d = 0.0;
    cuComplex A_c = make_cuComplex(1,0);
    cuDoubleComplex A_z = make_cuDoubleComplex(1,0);

    float B_f = 0;
    double B_d = 0.0;
    cuComplex B_c = make_cuComplex(1,0);
    cuDoubleComplex B_z = make_cuDoubleComplex(1,0);

    float D_f = 0;
    double D_d = 0.0;
    cuComplex D_c = make_cuComplex(1,0);
    cuDoubleComplex D_z = make_cuDoubleComplex(1,0);

    float E_f = 0;
    double E_d = 0.0;
    cuComplex E_c = make_cuComplex(1,0);
    cuDoubleComplex E_z = make_cuDoubleComplex(1,0);

    float TAU_f = 0;
    double TAU_d = 0.0;
    cuComplex TAU_c = make_cuComplex(1,0);
    cuDoubleComplex TAU_z = make_cuDoubleComplex(1,0);

    float TAUQ_f = 0;
    double TAUQ_d = 0.0;
    cuComplex TAUQ_c = make_cuComplex(1,0);
    cuDoubleComplex TAUQ_z = make_cuDoubleComplex(1,0);

    float TAUP_f = 0;
    double TAUP_d = 0.0;
    cuComplex TAUP_c = make_cuComplex(1,0);
    cuDoubleComplex TAUP_z = make_cuDoubleComplex(1,0);

    float C_f = 0;
    double C_d = 0.0;
    cuComplex C_c = make_cuComplex(1,0);
    cuDoubleComplex C_z = make_cuDoubleComplex(1,0);

    int lda = 0;
    int ldb = 0;
    const int ldc = 0;
    float workspace_f = 0;
    double workspace_d = 0;
    cuComplex workspace_c = make_cuComplex(1,0);
    cuDoubleComplex workspace_z = make_cuDoubleComplex(1,0);
    int Lwork = 0;
    int devInfo = 0;
    int devIpiv = 0;

    size_t b_size;

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potrf(**cusolverH, uplo, n, (float*)&A_f, lda, (float*)&workspace_f, Lwork), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potrf(**cusolverH, uplo, n, (double*)&A_d, lda, (double*)&workspace_d, Lwork), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potrf(**cusolverH, uplo, n, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&workspace_c, Lwork), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potrf(**cusolverH, uplo, n, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&workspace_z, Lwork), 0);
    status = cusolverDnSpotrf(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    status = cusolverDnDpotrf(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    status = cusolverDnCpotrf(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    status = cusolverDnZpotrf(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::potrs_scratchpad_size<float>(**cusolverH ,uplo ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: float *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<float>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::potrs(**cusolverH, uplo, n, nrhs, (float*)&C_f, lda, (float*)&B_f, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::potrs_scratchpad_size<double>(**cusolverH ,uplo ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: double *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<double>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::potrs(**cusolverH, uplo, n, nrhs, (double*)&C_d, lda, (double*)&B_d, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::potrs_scratchpad_size<std::complex<float>>(**cusolverH ,uplo ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: std::complex<float> *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<std::complex<float>>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::potrs(**cusolverH, uplo, n, nrhs, (std::complex<float>*)&C_c, lda, (std::complex<float>*)&B_c, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::potrs_scratchpad_size<std::complex<double>>(**cusolverH ,uplo ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: std::complex<double> *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<std::complex<double>>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::potrs(**cusolverH, uplo, n, nrhs, (std::complex<double>*)&C_z, lda, (std::complex<double>*)&B_z, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);
    cusolverDnDpotrs(*cusolverH, uplo, n, nrhs, &C_d, lda, &B_d, ldb, &devInfo);
    cusolverDnCpotrs(*cusolverH, uplo, n, nrhs, &C_c, lda, &B_c, ldb, &devInfo);
    cusolverDnZpotrs(*cusolverH, uplo, n, nrhs, &C_z, lda, &B_z, ldb, &devInfo);


    //CHECK: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrf_scratchpad_size<float>(**cusolverH ,m ,n ,lda);
    //CHECK-NEXT: mkl::lapack::getrf(**cusolverH, m, n, (float*)&A_f, lda, &result_temp_pointer6, (float*)&workspace_f, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrf_scratchpad_size<double>(**cusolverH ,m ,n ,lda);
    //CHECK-NEXT: mkl::lapack::getrf(**cusolverH, m, n, (double*)&A_d, lda, &result_temp_pointer6, (double*)&workspace_d, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrf_scratchpad_size<std::complex<float>>(**cusolverH ,m ,n ,lda);
    //CHECK-NEXT: mkl::lapack::getrf(**cusolverH, m, n, (std::complex<float>*)&A_c, lda, &result_temp_pointer6, (std::complex<float>*)&workspace_c, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrf_scratchpad_size<std::complex<double>>(**cusolverH ,m ,n ,lda);
    //CHECK-NEXT: mkl::lapack::getrf(**cusolverH, m, n, (std::complex<double>*)&A_z, lda, &result_temp_pointer6, (std::complex<double>*)&workspace_z, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: }
    cusolverDnSgetrf(*cusolverH, m, n, &A_f, lda, &workspace_f, &devIpiv, &devInfo);
    cusolverDnDgetrf(*cusolverH, m, n, &A_d, lda, &workspace_d, &devIpiv, &devInfo);
    cusolverDnCgetrf(*cusolverH, m, n, &A_c, lda, &workspace_c, &devIpiv, &devInfo);
    cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);


    //CHECK: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrs_scratchpad_size<float>(**cusolverH ,trans ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: float *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<float>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::getrs(**cusolverH, trans, n, nrhs, (float*)&A_f, lda, &result_temp_pointer6, (float*)&B_f, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrs_scratchpad_size<double>(**cusolverH ,trans ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: double *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<double>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::getrs(**cusolverH, trans, n, nrhs, (double*)&A_d, lda, &result_temp_pointer6, (double*)&B_d, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrs_scratchpad_size<std::complex<float>>(**cusolverH ,trans ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: std::complex<float> *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<std::complex<float>>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::getrs(**cusolverH, trans, n, nrhs, (std::complex<float>*)&A_c, lda, &result_temp_pointer6, (std::complex<float>*)&B_c, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer6;
    //CHECK-NEXT: std::int64_t scratchpad_size_ct{{[0-9]+}} = mkl::lapack::getrs_scratchpad_size<std::complex<double>>(**cusolverH ,trans ,n ,nrhs ,lda ,ldb);
    //CHECK-NEXT: std::complex<double> *scratchpad_ct{{[0-9]+}} = sycl::malloc_device<std::complex<double>>(scratchpad_size_ct{{[0-9]+}}, **cusolverH);
    //CHECK-NEXT: sycl::event event_ct{{[0-9]+}};
    //CHECK-NEXT: event_ct{{[0-9]+}} = mkl::lapack::getrs(**cusolverH, trans, n, nrhs, (std::complex<double>*)&A_z, lda, &result_temp_pointer6, (std::complex<double>*)&B_z, ldb, scratchpad_ct{{[0-9]+}}, scratchpad_size_ct{{[0-9]+}});
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer6;
    //CHECK-NEXT: std::vector<void *> ws_vec_ct{{[0-9]+}}{scratchpad_ct{{[0-9]+}}};
    //CHECK-NEXT: std::thread mem_free_thread(dpct::detail::mem_free, *cusolverH, ws_vec_ct{{[0-9]+}}, event_ct{{[0-9]+}});
    //CHECK-NEXT: mem_free_thread.detach();
    //CHECK-NEXT: }
    cusolverDnSgetrs(*cusolverH, trans, n, nrhs, &A_f, lda, &devIpiv, &B_f, ldb, &devInfo);
    cusolverDnDgetrs(*cusolverH, trans, n, nrhs, &A_d, lda, &devIpiv, &B_d, ldb, &devInfo);
    cusolverDnCgetrs(*cusolverH, trans, n, nrhs, &A_c, lda, &devIpiv, &B_c, ldb, &devInfo);
    cusolverDnZgetrs(*cusolverH, trans, n, nrhs, &A_z, lda, &devIpiv, &B_z, ldb, &devInfo);


    //CHECK: mkl::lapack::geqrf(**cusolverH, m, n, (float*)&A_f, lda, (float*)&TAU_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::geqrf(**cusolverH, m, n, (double*)&A_d, lda, (double*)&TAU_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::geqrf(**cusolverH, m, n, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::geqrf(**cusolverH, m, n, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSgeqrf(*cusolverH, m, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDgeqrf(*cusolverH, m, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnCgeqrf(*cusolverH, m, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZgeqrf(*cusolverH, m, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    //CHECK: mkl::lapack::ormqr(**cusolverH, side, trans, m, n, k, (float*)&A_f, lda, (float*)&TAU_f, (float*)&B_f, ldb, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::ormqr(**cusolverH, side, trans, m, n, k, (double*)&A_d, lda, (double*)&TAU_d, (double*)&B_d, ldb, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::unmqr(**cusolverH, side, trans, m, n, k, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&B_c, ldb, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::unmqr(**cusolverH, side, trans, m, n, k, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&B_z, ldb, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSormqr(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnDormqr(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnCunmqr(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnZunmqr(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);


    //CHECK: mkl::lapack::orgqr(**cusolverH, m, n, k, (float*)&A_f, lda, (float*)&TAU_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::orgqr(**cusolverH, m, n, k, (double*)&A_d, lda, (double*)&TAU_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::ungqr(**cusolverH, m, n, k, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::ungqr(**cusolverH, m, n, k, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSorgqr(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDorgqr(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnCungqr(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZungqr(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: int64_t result_temp_pointer5;
    //CHECK-NEXT: mkl::lapack::sytrf(**cusolverH, uplo, n, (float*)&A_f, lda, &result_temp_pointer5, (float*)&workspace_f, Lwork);
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer5;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer5;
    //CHECK-NEXT: mkl::lapack::sytrf(**cusolverH, uplo, n, (double*)&A_d, lda, &result_temp_pointer5, (double*)&workspace_d, Lwork);
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer5;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer5;
    //CHECK-NEXT: mkl::lapack::sytrf(**cusolverH, uplo, n, (std::complex<float>*)&A_c, lda, &result_temp_pointer5, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer5;
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: int64_t result_temp_pointer5;
    //CHECK-NEXT: mkl::lapack::sytrf(**cusolverH, uplo, n, (std::complex<double>*)&A_z, lda, &result_temp_pointer5, (std::complex<double>*)&workspace_z, Lwork);
    //CHECK-NEXT:  *&devIpiv = result_temp_pointer5;
    //CHECK-NEXT: }
    cusolverDnSsytrf(*cusolverH, uplo, n, &A_f, lda, &devIpiv, &workspace_f, Lwork, &devInfo);
    cusolverDnDsytrf(*cusolverH, uplo, n, &A_d, lda, &devIpiv, &workspace_d, Lwork, &devInfo);
    cusolverDnCsytrf(*cusolverH, uplo, n, &A_c, lda, &devIpiv, &workspace_c, Lwork, &devInfo);
    cusolverDnZsytrf(*cusolverH, uplo, n, &A_z, lda, &devIpiv, &workspace_z, Lwork, &devInfo);
}