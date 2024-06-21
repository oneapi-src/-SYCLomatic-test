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
import glob

from test_utils import *
user_root_path = ""
old_tilde = ""
cur_dir = ""
def setup_test():
    global old_tilde
    change_dir(test_config.current_test)
    cur_dir = os.getcwd()
    old_tilde = os.path.expanduser("~")
    os.environ["HOME"] = cur_dir
    return True

def migrate_test():
    global user_root_path
    user_root_path = os.path.expanduser("~") + "/workspace"
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    os.makedirs(user_root_path)
    distutils.dir_util.copy_tree(test_config.include_path + "/..", user_root_path, preserve_symlinks=True)
    change_dir("test")
    return call_subprocess("dpct vector_add.cu --cuda-include-path=" + "~/workspace/include")

def build_test():
    global old_tilde
    global user_root_path
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    os.environ["HOME"] = old_tilde
    return True

def run_test():
    return True

