# ====------ test_help.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os
import re
import sys
from pathlib import Path
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

def setup_test():
    return True

def migrate_test():
    return True

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    test_config.out_root = os.getcwd()

    lpthread_link_cases = ["devicemgr_multi_thread_aware", "memory_async_dpct_free", "blas_extension_api_usm",
                           "blas_extension_api_buffer", "fft_utils_engine_buffer", "fft_utils_engine_usm",
                           "cufft_workspace_interface", "fft_set_workspace"]
    oneDPL_related = ["thrust_test_device_ptr_2", "thrust_test-pennet_simple_pstl", "test_default_queue_2"]
    blas_cases = ["blas_utils_getrf", "blas_utils_getrf-usm", "blas_utils_getrf-complex",
                "blas_utils_getrf-complex-usm", "blas_utils_getrs", "blas_utils_getrs-usm",
                "blas_utils_getrs-complex", "blas_utils_getrs-complex-usm", "blas_utils_getri",
                "blas_utils_getri-usm", "blas_utils_getri-complex", "blas_utils_getri-complex-usm",
                "blas_utils_geqrf", "blas_utils_geqrf-usm", "blas_utils_geqrf-complex",
                "blas_utils_geqrf-complex-usm", "blas_utils_get_transpose", "blas_utils_get_value",
                "blas_utils_get_value_usm", "blas_extension_api_buffer", "lib_common_utils_mkl_get_version",
                "blas_extension_api_usm", "blas_utils_getrfnp", "blas_utils_getrfnp-usm", "blas_utils_getrfnp-complex",
                "blas_utils_getrfnp-complex-usm", "blas_utils_parameter_wrapper_buf", "blas_utils_parameter_wrapper_usm",
                "blas_utils_gels-usm"]
    oneDNN_related = ["dnnl_utils_activation", "dnnl_utils_fill", "dnnl_utils_lrn", "dnnl_utils_memory",
                "dnnl_utils_pooling", "dnnl_utils_reorder", "dnnl_utils_scale", "dnnl_utils_softmax",
                "dnnl_utils_sum", "dnnl_utils_reduction", "dnnl_utils_binary", "dnnl_utils_batch_normalization_1",
                "dnnl_utils_batch_normalization_2", "dnnl_utils_batch_normalization_3", "dnnl_utils_convolution_1",
                "dnnl_utils_convolution_2", "dnnl_utils_convolution_3", "dnnl_utils_convolution_4", "dnnl_utils_convolution_5",
                "dnnl_utils_normalization_1", "dnnl_utils_normalization_2", "dnnl_utils_normalization_3", "dnnl_utils_rnn",
                "dnnl_utils_version", "dnnl_utils_dropout", "blas_gemm_utils_interface"]
    fft_cases = ["fft_utils_engine_buffer", "fft_utils_engine_usm", "fft_workspace_interface", "fft_set_workspace"]
    lapack_cases = ["lapack_utils_buffer", "lapack_utils_usm"]
    rng_cases = ["rng_generator", "rng_generator_vec_size_1", "rng_host"]
    sparse_cases = ["sparse_utils_1", "sparse_utils_2_buffer", "sparse_utils_2_usm", "sparse_utils_3_buffer", "sparse_utils_3_usm", "sparse_utils_4_buffer", "sparse_utils_4_usm"]

    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    if test_config.current_test in oneDPL_related:
        cmp_opts.append(prepare_oneDPL_specific_macro())
    if test_config.current_test in lpthread_link_cases and platform.system() == "Linux":
        link_opts.append("-lpthread")

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        srcs.append(os.path.abspath(os.path.join(dirpath, test_config.current_test + ".cpp")))
    ret = False

    if test_config.current_test == "test_default_queue_2":
        srcs.append("test_default_queue_1.cpp")
    if test_config.current_test == "multi_definition_test":
        return True
    if test_config.current_test == "kernel_function_lin":
        ret = call_subprocess(test_config.DPCXX_COM + " -shared -fPIC -o module.so kernel_module_lin.cpp")
        if not ret:
            print("kernel_function_lin created the shared lib failed.")
            return False
    if test_config.current_test == "kernel_function_win":
        ret = call_subprocess("icx-cl -fsycl /EHsc /LD kernel_module_win.cpp /link /OUT:module.dll")
        if not ret:
            print("kernel_function_win created the shared lib failed.")
            return False

    if test_config.current_test in oneDNN_related:
        if platform.system() == 'Linux':
            link_opts.append(' -ldnnl')
        else:
            link_opts.append(' dnnl.lib')
    if (test_config.current_test in blas_cases) or (test_config.current_test in fft_cases) or (
       test_config.current_test in lapack_cases) or (test_config.current_test in rng_cases) or (
       test_config.current_test in oneDNN_related) or (test_config.current_test in sparse_cases):
        mkl_opts = []
        if test_config.test_option == 'option_cuda_backend':
            mkl_opts = ["-lonemath"]
        else:
            if platform.system() == "Linux":
                mkl_opts = test_config.mkl_link_opt_lin
            else:
                mkl_opts = test_config.mkl_link_opt_win

        link_opts += mkl_opts
        cmp_opts.append("-DMKL_ILP64")
    if test_config.current_test == 'fft_utils_engine_buffer' or test_config.current_test == 'fft_utils_engine_usm':
        ret = compile_and_link([os.path.join(test_config.out_root, 'cufft_test.dp.cpp')], cmp_opts, objects, link_opts)
    else:
        ret = compile_and_link(srcs, cmp_opts, objects, link_opts)
    return ret


def run_test():
    os.environ["ONEAPI_DEVICE_SELECTOR"] = test_config.device_filter
    args = []

    if test_config.current_test == "multi_definition_test":
        change_dir(test_config.current_test)
        test_driver = test_config.current_test + ".py"
        os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
        call_subprocess("python " + test_driver)
        if "case pass" in test_config.command_output:
            return True
        return False

    if test_config.current_test == "kernel_function_lin":
        args.append("./module.so")
    if test_config.current_test == "kernel_function_win":
        args.append("./module.dll")
    os.environ['CL_CONFIG_CPU_EXPERIMENTAL_FP16']="1"
    ret = run_binary_with_args(args)
    if test_config.current_test == "async_exception" and "Caught asynchronous SYCL exception" in test_config.command_output and "test_dpct_async_handler" in test_config.command_output:
        return True
    return ret

