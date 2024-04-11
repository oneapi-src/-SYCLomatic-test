# ====------ test_regression.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os
import re
import sys
import shutil
from pathlib import Path

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

def setup_test():
    return True

def migrate_test():
    if test_config.current_test in ["test-1399", "test-1247"]:
        os.chdir(test_config.current_test)
        return call_subprocess("sh ./run.sh")
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), test_config.current_test)

    test_config.out_root = os.path.join(in_root, 'out_root')
    # Clean the out-root when it exisits.
    if os.path.exists(test_config.out_root):
        shutil.rmtree(test_config.out_root)
    for dirpath, dirnames, filenames in os.walk(in_root):
        for filename in [f for f in filenames if re.match('.*(cu|cpp|c)$', f)]:
            src.append(os.path.abspath(os.path.join(dirpath, filename)))

    return do_migrate(src, in_root, test_config.out_root, extra_args)

def build_test():
    if test_config.current_test in ["test-1399", "test-1247"]:
        return True
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        for filename in [f for f in filenames if re.match('.*(cpp|c)$', f)]:
            srcs.append(os.path.abspath(os.path.join(dirpath, filename)))

    mkl_related_cases = ["test-1585", "test-1554", "test-1765", "test-1766a", "test-1766b", \
                "test-850a", "test-850b", "test-850c"]

    if test_config.current_test in mkl_related_cases:
        if platform.system() == 'Linux':
            link_opts = test_config.mkl_link_opt_lin
        else:
            link_opts = test_config.mkl_link_opt_win
    ret = False
    ret = compile_and_link(srcs, cmp_opts, objects, link_opts)
    return ret

def run_test():
    args = []
    if test_config.current_test in ["test-1399", "test-1247", "test-1662"]:
        return True
    if test_config.current_test == "test-1601":
        args.append("12 12 12")
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    return run_binary_with_args(args)
