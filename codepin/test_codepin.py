# ====------ test_codepin.py---------- *- Python -* ----===##
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

def find_file_with_keyword(directory, keyword):
    matching_files = []
    for file in os.listdir(directory):
        if keyword in file:
            matching_files.append(os.path.join(directory, file))
    return matching_files[0] # return the first matching file

def setup_test():
    return True

def migrate_test():
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
    in_root = os.path.join(os.getcwd(), test_config.current_test)
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    if "enable-codepin" in test_config.migrate_option:
        return build_codepin_cuda(in_root) and build_codepin_sycl(in_root)

def run_test():
    args = []
    ret = False
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    if "enable-codepin" in test_config.migrate_option:
        ret = run_codepin_cuda_and_sycl_binary(test_config.current_test + "_codepin_cuda.run", test_config.current_test + "_codepin_sycl.run")
        if ret == True:
            cuda_log = find_file_with_keyword(os.getcwd(), "CodePin_CUDA_")
            sycl_log = find_file_with_keyword(os.getcwd(), "CodePin_SYCL_")
            if compare_codepin_log(cuda_log, sycl_log) == 3:
                ret = True
    return ret