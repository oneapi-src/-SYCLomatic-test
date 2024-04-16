# ===------- do_test.py----------------------------------- *- Python -* ----===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===-----------------------------------------------------------------------===#

import subprocess
import platform
import os
import sys

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    call_subprocess(test_config.CT_TOOL + " --optimize-migration --out-root=./out test.cu --cuda-include-path=" + test_config.include_path)
    ret = ""
    with open(os.path.join("out", "test.dp.cpp"), 'r') as f:
        ret = f.read()
    if not is_sub_string("#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)", ret) and is_sub_string("asm(", ret):
        return False
    change_dir("out")
    return ret
def build_test():
    return True

def run_test():
    return True
