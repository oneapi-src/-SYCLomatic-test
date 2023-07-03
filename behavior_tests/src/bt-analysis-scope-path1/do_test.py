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
import shutil

from test_utils import *


def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True


def migrate_test(single_case_text):
    # clean previous migration output
    if (os.path.exists("out")):
        shutil.rmtree("out")
    migrate_cmd = test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " + os.path.join(
        "cuda",
        "call_device_func_outside.cu") + " --in-root=cuda" + " --out-root=out"
    # migrate with implicit --analysis-scope-path which defaults to --in-root
    call_subprocess(migrate_cmd, single_case_text)
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    shutil.rmtree("out")

    # migrate with specified --analysis-scope-path which equals --in-root
    call_subprocess(migrate_cmd + " --analysis-scope-path=cuda", single_case_text)
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    shutil.rmtree("out")

    # migrate with specified --analysis-scope-path which is the parent of --in-root
    call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                    os.path.join("cuda", ".."), single_case_text)
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    return True


def build_test(single_case_text):
    return True


def run_test(single_case_text):
    return True