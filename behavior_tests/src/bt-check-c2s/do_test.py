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
from hashlib import md5

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    dpct_path = shutil.which("dpct")
    c2s_path = shutil.which("c2s")

    print("dpct_path:", dpct_path)
    print("c2s_path:", c2s_path)

    subprocess.run(["ls", "-l", dpct_path])
    subprocess.run(["ls", "-l", c2s_path])

    dpct_file = open(dpct_path, "rb")
    c2s_file = open(c2s_path, "rb")

    dpct_md5 = md5(dpct_file.read()).hexdigest()
    c2s_md5 = md5(c2s_file.read()).hexdigest()
    print("dpct_md5:" + dpct_md5)
    print("c2s_md5:" + c2s_md5)

    dpct_file.close()
    c2s_file.close()
    return c2s_md5 == dpct_md5

def build_test():
    return True

def run_test():
    return True