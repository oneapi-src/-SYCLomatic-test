# ====------ test_utils.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
from genericpath import exists
from posixpath import isabs
import shutil
import distutils
import subprocess
from typing import DefaultDict
import xml.etree.ElementTree as ET
import os
import re
import platform
import sys
import test_config

import distutils

from distutils import dir_util

# Call subprocess to run migration, build and test binary. Store the command and execution result to
# command.tst and result.md.
def call_subprocess(cmd):
    test_config.executed_command += cmd +"\n"
    try:
        run_on_shell = False
        if (platform.system() == 'Linux'):
            run_on_shell = True
        complete_process = subprocess.run(cmd, shell=run_on_shell, check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", timeout=test_config.timeout)
        test_config.command_output += complete_process.stdout + "\n"
        test_config.execution_log += complete_process.stdout + "\n"
        if complete_process.returncode != 0:
            return False
        return True
    except subprocess.TimeoutExpired:
        test_config.execution_log += "========= Excuting " + cmd + " is meeting time out(" + str(test_config.timeout) + "s) Please check. ======"
        return False

def change_dir(dir):
    cmd = "cd " + dir

    with open(test_config.command_file, 'a+') as f:
        f.write(cmd + "\n")
    if os.path.exists(dir):
        os.chdir(dir)
    return True

def set_default_compiler(use_cuda : bool):
    if (use_cuda):
        test_config.DPCXX_COM = 'clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda'
        return
    if (platform.system() == 'Windows'):
        test_config.DPCXX_COM = "icx-cl -fsycl"
    else:
        test_config.DPCXX_COM = "icpx -fsycl"

def set_default_c_compiler(use_cuda : bool):
    if (use_cuda):
        test_config.C_COM = 'clang'
        return
    if (platform.system() == 'Windows'):
        test_config.C_COM = "icx-cc"
    else:
        test_config.C_COM = "icx"

def print_debug_log(desc, *args):
    if (test_config.VERBOSE_LEVEL != 0):
        print(desc + " : " )
        for arg in args:
            print(arg)
        print('\n')

def compile_files(srcs, cmpopts = []):
    ret = True
    for src in srcs:
        if os.path.splitext(src)[1] == '.c':
            base_cmd = test_config.C_COM + " -c "
        else:
            base_cmd = test_config.DPCXX_COM + " -c "
            if (platform.system() == 'Windows'):
                base_cmd += " /EHsc -DNOMINMAX "
        cmd = base_cmd + src + ' ' + ' '.join(cmpopts)
        ret = call_subprocess(cmd) and ret
    return ret

def prepare_obj_name(src):
    obj_name = re.split('\.', os.path.basename(src))
    suffix = 'o'
    if (platform.system() == 'Windows'):
      suffix = 'obj'
    obj_name[-1] = suffix
    return '.'.join(obj_name)

def compile_and_link(srcs, cmpopts = [], objects = [], linkopt = []):
    if not compile_files(srcs, cmpopts):
        return False
    obj_files = []
    for src in srcs:
        new_obj = prepare_obj_name(src)
        if new_obj not in obj_files:
            obj_files.append(new_obj)
    cmd = test_config.DPCXX_COM + ' '  + ' '.join(obj_files) + ' ' + \
                        ' '.join(linkopt) + ' ' + ' '.join(objects) + ' -o ' + test_config.current_test + '.run'
    return call_subprocess(cmd)


def run_binary_with_args(args = []):
    cmd = os.path.join(os.path.curdir, test_config.current_test + '.run ') + ' '.join(args)
    return call_subprocess(cmd)


# Replace the ${testName} with the specific test name.
def replace_test_name(test_name, source_file):
    return source_file.replace("${testName}", test_name)

# Copy the original source(source_files) to work space(test_ws_path).
def copy_source_to_ws(source_files, test_ws_path, suite_root_path):
    print_debug_log("work space path is ", test_ws_path)
    print_debug_log("source files is ", source_files)
    for file in source_files:
        print_debug_log("The file copied: ", file)
        if "${testName}" in file:
            file.replace("${testName}", test_config.current_test)
        file = os.path.join(suite_root_path, file)
        if os.path.isdir(file):
            distutils.dir_util.copy_tree(file, test_ws_path)
        else:
            shutil.copy(file, test_ws_path)
    return

def prepare_oneDPL_specific_macro():
    if (platform.system() == 'Windows'):
        return ''
    call_subprocess('gcc -dumpversion')
    if ('9' in test_config.command_output):
        return "-DPSTL_USE_PARALLEL_POLICIES=0"
    elif ('10' in test_config.command_output):
        return "-D_GLIBCXX_USE_TBB_PAR_BACKEND=0"
    return ""

def get_cuda_version():
    cuda_header = os.path.join(test_config.include_path, 'cuda.h')
    if not os.path.exists(cuda_header):
        exit("CUDA header file cannot be found.")
    with open(cuda_header, 'r') as f:
        for cnt, line in enumerate(f):
            if '#define CUDA_VERSION' in line:
                return int(line.split(' ')[-1])
    return 0

def append_msg_to_file(file_path, msg):
    with open(file_path, 'a+') as f:
        f.write(msg)

def do_migrate(src, in_root, out_root, extra_args = []):
    cmd = test_config.CT_TOOL  + " --cuda-include-path=" + test_config.include_path + \
            ' ' + ' '.join(src)
    if in_root:
        cmd += ' --in-root ' + os.path.abspath(in_root)
    if out_root:
        cmd += ' --out-root ' + out_root
    if extra_args:
        for arg in extra_args:
            cmd +=  ' --extra-arg=\" ' + arg + '\"'
    if test_config.migrate_option:
        cmd += ' ' + test_config.migrate_option
    return call_subprocess(cmd)

def check_migration_result(msg):
    with open(test_config.log_file, 'a+') as f:
        if msg in f.read():
            return True
    return False

# Check whether the test case has a specific test module or not.
# If the test case has the specific do_test.py test driver,
# the common driver execution will be skipped.
def is_registered_module(test_case_workspace):
    do_test_script = os.path.join(test_case_workspace, 'do_test.py')
    if os.path.exists(do_test_script):
        return True
    return False

# Print the failed test result and details in the screen.
def print_result(case, status, details_log):
    print("============= " + case + ": " + status + " ==================\n")
    print("========== Device Runtime Info: ===============")
    print(test_config.command_output)
    print("=============================================\n")
    print("----------------------------\n" + details_log + "\n----------------------\n")

def is_sub_string(substr, fullstr):
    if substr in fullstr:
        return True
    return False

# Get dpct path, None if dpct is not in $PATH
def get_ct_path():
    return shutil.which('dpct')

# Get the CT's bundled clang version
# E.g., <dpct_root>/bin/dpct/../../lib/clang/15.0.0
def get_ct_clang_version():
    ct_path = get_ct_path()
    if ct_path:
        ct_clang_path = os.path.join(os.path.dirname(ct_path), "..", "lib", "clang")
        sub_dirs = [dir.name for dir in os.scandir(ct_clang_path) if dir.is_dir() and re.fullmatch("(\d+).*", dir.name)]
        if len(sub_dirs) > 0:
            sub_dirs.sort()
            return sub_dirs[-1] + ".0.0"
    return None
