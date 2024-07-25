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
    os.mkdir("dpct_output")
    with open(os.path.join("dpct_output", "test"), 'w') as f:
        f.write("2")
    call_subprocess(test_config.CT_TOOL + " vector_add.cu  --cuda-include-path=" + test_config.include_path)
    return is_sub_string("directory is not empty.", test_config.command_output)

def build_test():
    return True

def run_test():
    return True
