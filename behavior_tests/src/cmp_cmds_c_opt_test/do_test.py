# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import filecmp
import subprocess
import platform
import os
import sys
import json

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def parse_compilation_database(compilation_database_name):
    if (not os.path.isfile(compilation_database_name)):
        print("The compilation database is not existed, please double check\n")

    with open(compilation_database_name) as compilation_database:
        compilation_database_items = json.load(compilation_database)
        database_source_files = []
        for entry in compilation_database_items:
            compilation_item = entry['file']
            database_source_files.append(os.path.basename(compilation_item))
        return database_source_files


def migrate_test():
    call_subprocess("intercept-build make -f Makefile CC=nvcc USE_SM=70 -B")
    test_dir = os.path.join(os.getcwd())
    print(test_dir)
    ref_cmp_db_file = open(test_dir+"/compile_commands.json_ref", "rt")
    cmp_cmds = ref_cmp_db_file.read()
    cmp_cmds = cmp_cmds.replace('${TEST_DIRECTORY}', test_dir)
    ref_cmp_db_file.close()
    ref_cmp_db_file = open(test_dir+"/compile_commands.json_ref", "wt")
    ref_cmp_db_file.write(cmp_cmds)
    ref_cmp_db_file.close()

    result = filecmp.cmp(test_dir+"/compile_commands.json",
                         test_dir+"/compile_commands.json_ref", shallow=False)

    return result


def build_test():
    return True


def run_test():
    return True
