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
from test_config import CT_TOOL

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    data = []
    ret = []
    with open("compile_commands.json", 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.replace("directory_placeholder", os.getcwd().replace("\\", "\\\\"))
            ret.append(line)
    with open("compile_commands.json", 'w') as f:
        f.writelines(ret)

    call_subprocess(test_config.CT_TOOL + ' -p=./ --cuda-include-path=' + test_config.include_path)
    return is_sub_string("The file name(s) in the \"command\" and \"file\" fields of the compilation database are inconsistent", test_config.command_output)


def build_test():
    return True

def run_test():
    return True

