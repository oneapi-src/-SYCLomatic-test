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

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    call_subprocess(test_config.CT_TOOL + " test.cu --in-root . --out-root out --always-use-async-handler --assume-nd-range-dim=1 --comments --enable-ctad --no-dpcpp-extensions=enqueued_barriers --no-dry-pattern --process-all -p . --sycl-named-lambda --use-experimental-features=free-function-queries,nd_range_barrier --use-explicit-namespace=sycl,dpct --usm-level=none --cuda-include-path=" + test_config.include_path)
    call_subprocess(test_config.CT_TOOL + " test.cu --out-root out --cuda-include-path=" + test_config.include_path)
    return is_sub_string("\"--analysis-scope-path=\"", test_config.command_output) and \
           is_sub_string("--always-use-async-handler --comments --compilation-database=\"", test_config.command_output) and \
           is_sub_string("--enable-ctad --use-experimental-features=free-function-queries,nd_range_barrier --use-explicit-namespace=sycl,dpct --no-dpcpp-extensions=enqueued_barriers --assume-nd-range-dim=1 --no-dry-pattern --process-all --sycl-named-lambda --usm-level=none\".", test_config.command_output)

def build_test():
    return True

def run_test():
    return True