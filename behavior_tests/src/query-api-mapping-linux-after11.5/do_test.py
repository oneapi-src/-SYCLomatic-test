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


def migrate_test():
    test_cases = [
        [  # CUB
            "cub::DeviceReduce::Max",
            [
                "cudaStream_t stream;",
                "cudaStreamCreate(&stream);",
                "cub::DeviceReduce::Max(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);",
            ],
            [],
            [
                "dpct::queue_ptr stream;",
                "stream = dpct::get_current_device().create_queue();",
                "stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::maximum<>()), 1).wait();",
            ],
        ],
    ]

    res = True
    for test_case in test_cases:
        res = res and test_api(test_case[0], test_case[1], test_case[2], test_case[3])
    return res


def build_test():
    return True


def run_test():
    return True
