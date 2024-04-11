# ===------ test_multi_devices.py ------------------------ *- Python -* ----===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

import os
import re
import sys
from pathlib import Path
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

def setup_test():
    return True

def migrate_test():
    return True

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    test_config.out_root = os.getcwd()

    mkl_related = ["rng"]

    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        srcs.append(os.path.abspath(os.path.join(dirpath, test_config.current_test + ".cpp")))

    if test_config.current_test in mkl_related:
        mkl_opts = []
        if platform.system() == "Linux":
            mkl_opts = test_config.mkl_link_opt_lin
        else:
            mkl_opts = test_config.mkl_link_opt_win
        link_opts += mkl_opts
        cmp_opts.append("-DMKL_ILP64")

    ret = compile_and_link(srcs, cmp_opts, objects, link_opts)
    return ret

def run_test():
    args = []
    ret = run_binary_with_args(args)
    return ret
