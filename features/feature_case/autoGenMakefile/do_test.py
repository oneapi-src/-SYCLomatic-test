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
    if (os.path.exists("out1")):
        shutil.rmtree("out1")

    if (os.path.exists("out2")):
        shutil.rmtree("out2")

    if (os.path.exists("out3")):
        shutil.rmtree("out3")

    ret = call_subprocess("intercept-build /usr/bin/make -B")
    if not ret:
        print("Error to create compilation database:", test_config.command_output)
    ret = call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " -in-root=.  -out-root=./out1 --gen-build-script -p ./")
    if not ret:
        print("Error to migration:", test_config.command_output)
    ret = call_subprocess("cd out1 && make -f Makefile.dpct")
    if not ret:
        print("Error to build:", test_config.command_output)
    ret1 =  os.path.exists("out1/runfile")
    if not ret:
        print("Error to check out1/runfile:", test_config.command_output)

    ret = call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " -in-root=.  -out-root=./out2 --gen-build-script -p ./ --build-script-file=Makefile")
    if not ret:
        print("Error to migration:", test_config.command_output)
    ret = call_subprocess("cd out2 && make")
    if not ret:
        print("Error to build:", test_config.command_output)
    ret2 =  os.path.exists("out2/runfile")
    if not ret2:
        print("Error to check out2/runfile:", test_config.command_output)


    ret = call_subprocess("make clean && rm compile_commands.json && intercept-build /usr/bin/make -f Makefile_mt")
    if not ret:
        print("Error to create compilation database:", test_config.command_output)
    ret = call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " -in-root=.  -out-root=./out3 --gen-build-script -p ./")
    if not ret:
        print("Error to migration:", test_config.command_output)
    ret = call_subprocess("cd out3 && make -f Makefile.dpct")
    if not ret:
        print("Error to build:", test_config.command_output)
    ret3 =  os.path.exists("out3/targets/runfile1")
    if not ret3 :
        print("Error to check out3/targets/runfile1:", test_config.command_output)
    ret4 =  os.path.exists("out3/targets/runfile2")
    if not ret4 :
        print("Error to check out3/targets/runfile2:", test_config.command_output)

    return ret1 and ret2 and ret3 and ret4
def build_test():
    return True
def run_test():
    return True
