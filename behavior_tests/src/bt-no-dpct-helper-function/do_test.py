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
    call_subprocess(test_config.CT_TOOL + " --out-root=./sycl test.cu --cuda-include-path=" + test_config.include_path)
    return True
def build_test():
    with open("./sycl/test.dp.cpp", "r") as f:
        # check migrated code content
        content = f.read()
        if "dpct::" in content:
            print("the migrated code should not contain 'dpct::':")
            print(content)
            print("case fail")
            return False
        else:
            print("case pass")
    return True

def run_test():
    return True
    