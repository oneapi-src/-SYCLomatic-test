# ===------------------- do_test.py ---------- *- Python -* ----------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===-----------------------------------------------------------------------===#

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def check(expect):
    if expect not in test_config.command_output:
        print("'", expect, "' is not int the output")
        return False
    return True


def test_auto_complete(api_name):
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping=-"
    )
    if api_name not in test_config.command_output:
        print("error message check failed:\n", api_name)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    print("auto complete check passed:", api_name)
    return True


def test_api(api_name, source_code, options, migrated_code):
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping="
        + api_name
    )
    ret = check("CUDA API:")
    for code in source_code:
        ret &= check(code)
    expect = "Is migrated to"
    if options.__len__() > 0:
        expect += " (with the option"
        for option in options:
            expect += " " + option
        expect += ")"
    expect += ":"
    ret &= check(expect)
    for code in migrated_code:
        ret &= check(code)
    if not ret:
        print("API query message check failed: ", api_name)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    print("API query message check passed: ", api_name)
    return True


def test_error():
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping=aaa"
    )
    expect = "dpct exited with code: -43 (Error: The API mapping query for this API is not available yet. You may get the API mapping by migrating sample code from this CUDA API to the SYCL API with the tool.)"
    if expect not in test_config.command_output:
        print("error message check failed:\n", expect)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping=hdiv"
    )
    expect = "dpct exited with code: -44 (Error: Can not find 'hdiv' in current CUDA header file: "
    if expect not in test_config.command_output:
        print("error message check failed:\n", expect)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    call_subprocess(
        test_config.CT_TOOL + " --cuda-include-path=." + " --query-api-mapping=hdiv"
    )
    expect = "dpct exited with code: -45 (Error: Cannot find 'hdiv' in current CUDA header file: "
    if expect not in test_config.command_output:
        print("error message check failed:\n", expect)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    print("error message check passed")
    return True


def migrate_test():
    test_cases = [
        [  # Runtime
            "cudaEventDestroy",
            ["cudaEventDestroy(e /*cudaEvent_t*/);"],
            [],
            ["dpct::destroy_event(e);"],
        ],
        [  # Driver
            "cuDeviceGetName",
            ["cuDeviceGetName(pc /*char **/, i /*int*/, d /*CUdevice*/);"],
            [],
            [
                "memcpy(pc, dpct::dev_mgr::instance().get_device(d).get_info<sycl::info::device::name>().c_str(), i);"
            ],
        ],
        [  # Math
            "__vaddss4",
            ["__vaddss4(u1 /*unsigned int*/, u2 /*unsigned int*/);"],
            ["--use-dpcpp-extensions=intel_device_math"],
            ["sycl::ext::intel::math::vaddss4(u1, u2);"],
        ],
        [  # cuBLAS
            "cublasSgemm",
            [
                "cublasSgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,",
                "            transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,",
                "            alpha /*const float **/, a /*const float **/, lda /*int*/,",
                "            b /*const float **/, ldb /*int*/, beta /*const float **/,",
                "            c /*float **/, ldc /*int*/);",
            ],
            [],
            [
                "oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc, dpct::blas::deduce_compute_mode(std::nullopt, handle->get_math_mode(), false));"
            ],
        ],
        [  # cuDNN
            "cudnnCreate",
            ["cudnnHandle_t h;", "cudnnCreate(&h /*cudnnHandle_t **/);"],
            [],
            ["dpct::dnnl::engine_ext h;"],
        ],
        [  # cuFFT
            "cufftCreate",
            ["cufftCreate(plan /*cufftHandle **/);"],
            [],
            ["*plan = dpct::fft::fft_engine::create();"],
        ],
        [  # cuRAND
            "curandCreateGenerator",
            [
                "curandCreateGenerator(pg /*curandGenerator_t **/, r /*curandRngType_t*/);"
            ],
            [],
            ["*(pg) = dpct::rng::create_host_rng(r);"],
        ],
        [  # cuSolver
            "cusolverDnCpotrfBatched",
            [
                "cusolverDnCpotrfBatched(handle /*cusolverDnHandle_t*/,",
                "                        upper_lower /*cublasFillMode_t*/, n /*int*/,",
                "                        a /*cuComplex ***/, lda /*int*/, info /*int **/,",
                "                        group_count /*int*/);",
            ],
            [],
            [
                "dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);"
            ],
        ],
        [  # cuSPARSE
            "cusparseSpMM",
            [
                "cusparseSpMM(handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,",
                "             transb /*cusparseOperation_t*/, alpha /*const void **/,",
                "             a /*cusparseSpMatDescr_t*/, b /*cusparseDnMatDescr_t*/,",
                "             beta /*const void **/, c /*cusparseDnMatDescr_t*/,",
                "             computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,",
                "             workspace /*void **/);",
            ],
            [],
            [
                "dpct::sparse::spmm(*handle, transa, transb, alpha, a, b, beta, c, computetype);"
            ],
        ],
        [  # NCCL
            "ncclGetUniqueId",
            ["ncclGetUniqueId(uniqueId /*ncclUniqueId **/);"],
            [],
            ["*uniqueId = dpct::ccl::create_kvs_address();"],
        ],
    ]

    res = True
    for test_case in test_cases:
        res = res and test_auto_complete(test_case[0])
        res = res and test_api(test_case[0], test_case[1], test_case[2], test_case[3])
    res = res and test_error()
    return res


def build_test():
    return True


def run_test():
    return True
