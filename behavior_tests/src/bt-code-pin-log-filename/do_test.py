# ====------ do_test.py----------- *- Python -* ----===##
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
import glob

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    ret_file = ""
    call_subprocess(test_config.CT_TOOL + " --enable-codepin test.cu --out-root=out --cuda-include-path=" + test_config.include_path)
    return True

def build_test():
    srcs = []
    srcs.append(os.path.join("out_codepin_sycl", "test.dp.cpp"))
    return compile_and_link(srcs)

def run_test():
    run_binary_with_args()
    matching_files = glob.glob("CodePin_SYCL_" + "*")
    if matching_files:
        return True
    return False
