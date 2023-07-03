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

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)

    return True

def migrate_test(single_case_text):
    file_vcxproj = os.path.join(os.getcwd(), "cuda", "file_c.vcxproj")
    migrated_file = os.path.join(os.getcwd(), "cuda", "file.c")
    expected_str = ""
    migrated_str = ""
    call_subprocess(single_case_text.CT_TOOL + " --report-file-prefix=report --out-root=out --vcxprojfile=\"" + \
            file_vcxproj +  "\" \"" + migrated_file + "\" --cuda-include-path=" + \
                   os.environ['CUDA_INCLUDE_PATH'], single_case_text)
    with open("expected.cpp", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "//" not in line:
                print(line)
                expected_str += line.strip()
    with open(os.path.join("out", "file.c.dp.cpp")) as f:
        lines = f.readlines()
        for line in lines:
            migrated_str += line.strip()
    print(expected_str)
    if expected_str in migrated_str:
        return True
    return False

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True