# ====------ do_test.py---------- *- Python -* ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
import os
import tempfile

from test_utils import *
from test_config import CT_TOOL


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    """
    Runs dpct with options for --sycl-file-extension and verfies the exisitance
    of migrated files with correct extension
    """
    # list of supported options mapped to input and output files
    options_and_result = get_sycl_file_extension_options_and_results()

    # To use absolute path on --in-root requires the same path in a pre-build
    # compile command. It is convenient to use ./ as we already change
    # directory to the parent dir of this script.
    cmd = f"{test_config.CT_TOOL} --in-root . --out-root {{0}} --cuda-include-path {test_config.include_path} -p ./compile_commands.json"

    result = True
    # Run default and supported options and test results
    for option, results in options_and_result.items():
        result &= run_test_for_option(cmd, option, results)

    # Check error message in case an unsupported value is passed to
    # --sycl-file-extension
    with tempfile.TemporaryDirectory() as dpct_out_root:
        cmd = f"{cmd} --sycl-file-extension=unsupported-option"
        call_subprocess(cmd)
        result &= is_sub_string(
            "Cannot find option named 'unsupported-option'", test_config.command_output
        )

    return result


def build_test():
    return True


def run_test():
    return True


def run_test_for_option(cmd, option, results):
    """
    The option is for --sycl-file-extension and results is a dict containing input file and output files.
    """
    if option:
        cmd = f"{cmd} --sycl-file-extension {option}"

    with tempfile.TemporaryDirectory() as dpct_out_root:
        # add temp dir path as dpct output file
        cmd = cmd.format(dpct_out_root)
        ret = call_subprocess(cmd)

        if not ret:
            print(f"user-defined-sycl-file-extension: dpct command: {cmd}")
            print(f"dpct output:\n {test_config.command_output}")
            return False

        for _, output_filename in results.items():
            out_filepath = os.path.join(dpct_out_root, output_filename)
            if not os.path.exists(out_filepath):
                print(
                    f"user-defined-sycl-file-extension: missing expected file: {out_filepath}"
                )
                print(f"dpct command: {cmd}")
                print(f"dpct output:\n {test_config.command_output}")
                return False

    return True


def get_sycl_file_extension_options_and_results():
    """
    Returns a dict of supported options mapped to input and output files
    """
    return {
        "dp-cpp": {
            "main.cu": "main.dp.cpp",
            "cuda_src.cu": "cuda_src.dp.cpp",
            "cuda_src.cuh": "cuda_src.dp.hpp",
            "src.cpp": "src.cpp",
            "src.h": "src.h",
        },
        "sycl-cpp": {
            "main.cu": "main.sycl.cpp",
            "cuda_src.cu": "cuda_src.sycl.cpp",
            "cuda_src.cuh": "cuda_src.sycl.hpp",
            "src.cpp": "src.cpp",
            "src.h": "src.h",
        },
        "cpp": {
            "main.cu": "main.cpp",
            "cuda_src.cu": "cuda_src.cpp",
            "cuda_src.cuh": "cuda_src.hpp",
            "src.cpp": "src.cpp",
            "src.h": "src.h",
        },
        "": {  # default case
            "main.cu": "main.dp.cpp",
            "cuda_src.cu": "cuda_src.dp.cpp",
            "cuda_src.cuh": "cuda_src.dp.hpp",
            "src.cpp": "src.cpp",
            "src.h": "src.h",
        },
    }
