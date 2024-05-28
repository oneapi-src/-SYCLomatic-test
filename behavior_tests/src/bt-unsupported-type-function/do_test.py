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
    migrated_file = os.path.join("out_nvml", "test.dp.cpp")
    call_subprocess(test_config.CT_TOOL + " test.cu --out-root=./out_nvml --cuda-include-path=" + test_config.include_path)
    warn_1007_count = 0
    warn_1082_count = 0
    with open(migrated_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "DPCT1007" in line:
                warn_1007_count += 1
            if "DPCT1082" in line:
                warn_1082_count += 1
    if warn_1007_count == 5 and warn_1082_count == 11:
        return True
    return False
def build_test():
    return True

def run_test():
    return True