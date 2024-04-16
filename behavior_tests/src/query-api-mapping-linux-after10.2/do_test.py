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
        [  # Thrust
            "thrust::uninitialized_copy_n",
            [
                "/*1*/ thrust::uninitialized_copy_n(d_input.begin() /*InputIterator*/, N /*Size*/,",
                "                             d_array /*ForwardIterator*/);",
                "/*2*/ thrust::uninitialized_copy_n(h_data /*InputIterator*/, N /*Size*/,",
                "                             h_array /*ForwardIterator*/);",
                "/*3*/ thrust::uninitialized_copy_n(",
                "    thrust::device /*const thrust::detail::execution_policy_base<",
                "                      DerivedPolicy > &*/,",
                "    d_input.begin() /*InputIterator*/, N /*Size*/,",
                "    d_array /*ForwardIterator*/);",
                "/*4*/ thrust::uninitialized_copy_n(",
                "    thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy",
                "                    > &*/,",
                "    h_data /*InputIterator*/, N /*Size*/, h_array /*ForwardIterator*/);",
            ],
            [],
            [
                "/*1*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);",
                "/*2*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);",
                "/*3*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);",
                "/*4*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);",
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
