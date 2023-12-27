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
    # clean previous migration output
    if (os.path.exists("build")):
        shutil.rmtree("build")    
    call_subprocess(" mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=icpx ../ && make")
    return os.path.exists(os.path.join("build", "app"))
def build_test():
    return True
def run_test():
    change_dir("build")
    return call_subprocess(os.path.join(os.path.curdir, "app"))
