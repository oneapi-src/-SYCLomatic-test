# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os

from test_utils import *
from test_config import CT_TOOL

rel_bin_path = "./build/run"


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    """
    Builds the project using cmake. The cmake list contains
    dpct_helper_sycl_compile.
    """
    # clean previous migration output
    if os.path.exists("build"):
        shutil.rmtree("build")

    # configure and build cmake containing dpct_helper_sycl_compile
    config_cmd = 'cmake -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=icpx -B build -S .'
    ret = call_subprocess(config_cmd)
    if not ret:
        print(f"Command '{config_cmd}' failed:", test_config.command_output)
        return False

    build_cmd = "cmake --build build --parallel 4"
    ret = call_subprocess(build_cmd)
    if not ret:
        print(f"Command '{build_cmd}' failed:", test_config.command_output)
        return False

    # make sure the binary exists
    if not os.path.exists(rel_bin_path):
        print(f"Failed to find {rel_bin_path} in {os.getcwd()}")
        return False

    return True


def build_test():
    return True


def run_test():
    """
    Run the binary and expect $? as zero
    """
    ret = call_subprocess(rel_bin_path)
    if not ret:
        print(
            f"Command '{rel_bin_path}' returned non-zero error code:",
            test_config.command_output,
        )
    return ret
