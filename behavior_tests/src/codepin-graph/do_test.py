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
import importlib.util
import shutil

from test_utils import *

def read_os_release():
    try:
        f = open('/etc/os-release', 'r')
        content = f.read()
        f.close()
        return content
    except FileNotFoundError:
        return None

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    osinfo = read_os_release()
    if osinfo is not None:
      if ("Ubuntu" in osinfo) and ("22.04" in osinfo):
        call_subprocess("codepin-report.py --instrumented-cuda-log=./src/CodePin_CUDA.json --instrumented-sycl-log=./src/CodePin_SYCL.json --generate-data-flow-graph")
        if os.path.exists("./CodePin_DataFlowGraph.pdf"):
          return True
        print("Expected file CodePin_DataFlowGraph.pdf is not exists.")
        return False
    return True

def build_test():
    return True

def run_test():
    return True