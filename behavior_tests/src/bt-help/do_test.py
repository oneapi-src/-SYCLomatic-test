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

def compare_file_with_string(file_path, input_string):
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Compare the file content with the input string
        if file_content == input_string:
            return True

    except FileNotFoundError:
        print(f'Error: The file "{file_path}" does not exist.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

    return False

def check_help_output(help_opts):
    call_subprocess(test_config.CT_TOOL + ' --help')
    if not compare_file_with_string('help_all.txt', test_config.command_output):
        print('Output mismtach for option: --help')
        return False

    for opt in help_opts:
        call_subprocess(test_config.CT_TOOL + ' --help=' + opt)
        if not compare_file_with_string('help_' + opt + '.txt', test_config.command_output):
            print('Output mismtach for option: --help=' + opt)
            return False

    return True

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    help_opts = ['advanced', 'basic']
    return check_help_output(help_opts)

def build_test():
    return True

def run_test():
    return True