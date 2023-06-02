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
    return call_subprocess(test_config.CT_TOOL + " --extra-arg=-I\"" + os.path.dirname(__file__) + 
                           "/header1 "+",-I" + os.path.dirname(__file__) + "/header2\"" + 
                           " --cuda-include-path=" + test_config.include_path + " test.cu")

def build_test():
    return True

def run_test():
    return True