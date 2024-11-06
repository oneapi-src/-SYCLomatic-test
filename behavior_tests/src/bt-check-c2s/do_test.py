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

    if platform.system() == 'Linux':
        dpct_file = open(dpct_path, "rb")
        c2s_file = open(c2s_path, "rb")
        dpct_md5 = md5(dpct_file.read()).hexdigest()
        c2s_md5 = md5(c2s_file.read()).hexdigest()
        print("dpct_md5:" + dpct_md5)
        print("c2s_md5:" + c2s_md5)
        dpct_file.close()
        c2s_file.close()
        return c2s_md5 == dpct_md5
    elif platform.system() == 'Windows':
        os.makedirs('copy_binary')
        shutil.copy(dpct_path, './copy_binary')
        shutil.copy(c2s_path, './copy_binary')
        subprocess.check_call(['signtool', 'remove', '/s', './copy_binary/dpct.exe'])
        subprocess.check_call(['signtool', 'remove', '/s', './copy_binary/c2s.exe'])
        dpct_file = open('./copy_binary/dpct.exe', "rb")
        c2s_file = open('./copy_binary/c2s.exe', "rb")
        dpct_md5 = md5(dpct_file.read()).hexdigest()
        c2s_md5 = md5(c2s_file.read()).hexdigest()
        print("dpct_md5:" + dpct_md5)
        print("c2s_md5:" + c2s_md5)
        dpct_file.close()
        c2s_file.close()
        shutil.rmtree('./copy_binary')
        return c2s_md5 == dpct_md5

def build_test():
    return True

def run_test():
    return True