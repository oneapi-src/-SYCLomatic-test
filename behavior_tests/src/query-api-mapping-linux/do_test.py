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


def test_color():
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping=cudaDeviceSynchronize"
    )
    expect = (
        "CUDA API:[0;32m\n"
        + "  cudaDeviceSynchronize();\n"
        + "[0mIs migrated to:[0;34m\n"
        + "  dpct::get_current_device().queues_wait_and_throw();\n"
        + "[0m"
    )
    if expect not in test_config.command_output:
        print("color output check failed:\n", expect)
        print("output:\n", test_config.command_output, "===end===\n")
        return False
    print("color output check passed")
    return True


def migrate_test():
    res = True
    res = res and test_color()
    return res


def build_test():
    return True


def run_test():
    return True
