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
    call_subprocess(test_config.CT_TOOL + " --helper-function-dir")
    helper_function_dir_root = os.path.realpath(
        os.path.join(get_ct_path(), os.pardir, os.pardir, "include"))
    print("Helper functions directory: ", helper_function_dir_root)
    if (platform.system() == 'Windows'):
        return is_sub_string(helper_function_dir_root.lower(), test_config.command_output.lower())
    return is_sub_string(helper_function_dir_root, test_config.command_output)

def build_test():
    return True

def run_test():
    return True
