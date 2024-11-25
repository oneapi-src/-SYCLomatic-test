# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import subprocess
import platform
import os
import sys

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    call_subprocess(
        test_config.CT_TOOL + " test.cu -analysis-mode --cuda-include-path=" + test_config.include_path)


    reference_lines = [
                        ["test.cu:"],
                        ["lines of code", "will be automatically migrated."],
                        ["APIs/Types - No manual effort."],
                        ["APIs/Types - Low manual effort for checking and code fixing."],
                        ["APIs/Types - Medium manual effort for code fixing."],
                        ["lines of code", "will not be automatically migrated."],
                        ["APIs/Types - High manual effort for code fixing."],
                        ["Total Project:"],
                        ["lines of code", "will be automatically migrated."],
                        ["APIs/Types - No manual effort."],
                        ["APIs/Types - Low manual effort for checking and code fixing."],
                        ["APIs/Types - Medium manual effort for code fixing."],
                        ["lines of code", "will not be automatically migrated."],
                        ["APIs/Types - High manual effort for code fixing."],
                        ["Library Dependencies of SYCL Project:"],
                        ["oneAPI Math Kernel Library (oneMKL)"],
                        ["oneAPI DPC++ Library (oneDPL)"],
                        ["oneAPI Deep Neural Network Library (oneDNN)"],
                        ["oneAPI Collective Communications Library (oneCCL)"]]
    
    ret_lines = test_config.command_output.splitlines()
    res = True
    idx = 0
    check_flag = False
    for ret in ret_lines:
        if check_flag:
            for ref in reference_lines[idx]:
                if ref not in ret:
                    res = False
                    print("there should be a '" + ref + "' in output '" + ret + "'.")
            idx = idx + 1
            if idx >= len(reference_lines):
                check_flag = False
        elif ret.startswith("Migrating:") and ret.endswith("test.cu"):
            check_flag = True

    return res


def build_test():
    return True


def run_test():
    return True
