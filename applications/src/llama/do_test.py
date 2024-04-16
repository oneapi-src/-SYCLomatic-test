# ===------- do_test.py ---------------------------------- *- Python -* ---=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

import os
import re
import sys
from pathlib import Path
from test_utils import *

def setup_test():
    return True

def migrate_test():
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), test_config.current_test)
    test_config.out_root = os.path.join(in_root, 'out_root')

    if test_config.current_test == 'llama' or test_config.current_test == 'llama_cmake_migration':
        original_string = '/export/users/placeholder/project/llama.cpp'
        new_string = in_root
        db_dir = os.path.join(in_root, 'gpubuild')
        with open(os.path.join(db_dir, 'compile_commands.json'), "r") as file:
            file_contents = file.read()
        file_contents = file_contents.replace(original_string, new_string)
        db_dir = db_dir.replace('\\', '\\\\')
        file_contents = re.sub(r'"directory": ".*?"', '"directory": "' + db_dir + '"', file_contents)
        file_contents = file_contents.replace('\\', '\\\\')
        with open(os.path.join(db_dir, 'compile_commands.json'), "w") as file:
            file.write(file_contents)
        src.append(' -p=' + os.path.join(in_root, 'gpubuild'))
        src.append(' --enable-profiling ')
        src.append(' --use-experimental-features=free-function-queries,local-memory-kernel-scope-allocation ')

        if test_config.current_test == 'llama_cmake_migration':
            src.append(' --migrate-build-script=CMake ')

    return do_migrate(src, in_root, test_config.out_root, extra_args)

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    llama_files = ['common.cpp', 'sampling.cpp', 'console.cpp', 'grammar-parser.cpp', 'train.cpp',
                   'build-info.cpp', 'llama.cpp', 'ggml.c', 'ggml-alloc.c', 'ggml-backend.c', 'ggml-quants.c',
                   'ggml-cuda.dp.cpp', 'main.cpp']

    if test_config.current_test == 'llama':
        if platform.system() == 'Linux':
            link_opts = test_config.mkl_link_opt_lin
        else:
            link_opts = test_config.mkl_link_opt_win
        cmp_opts.append("-DMKL_ILP64")
        cmp_opts.append("-DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128")
        cmp_opts.append("-DGGML_USE_CUBLAS -DK_QUANTS_PER_ITERATION=2 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -O3 -DNDEBUG")
        cmp_opts.append("-I " + test_config.out_root)
        cmp_opts.append("-I " + os.path.join(test_config.out_root, 'common'))

        for dirpath, dirnames, filenames in os.walk(test_config.out_root):
            if test_config.current_test == 'llama':
                for filename in filenames:
                    if filename in llama_files:
                        srcs.append(os.path.abspath(os.path.join(dirpath, filename)))
            else:
                for filename in [f for f in filenames if re.match('.*(cpp|c)$', f)]:
                    srcs.append(os.path.abspath(os.path.join(dirpath, filename)))

        if platform.system() == 'Linux':
            link_opts.append(' -lpthread ')

        ret = False

    if test_config.current_test == 'llama':
        ret = compile_and_link(srcs, cmp_opts, objects, link_opts)
    elif test_config.current_test == 'llama_cmake_migration':
        ret = call_subprocess("cp common/base64.hpp ./out_root/common/base64.hpp")
        ret += call_subprocess("cp common/build-info.cpp.in  ./out_root/common/build-info.cpp.in")
        ret += call_subprocess("cp scripts/build-info.sh out_root/scripts/build-info.sh")
        ret += call_subprocess("cp scripts/build-info.h.in out_root/scripts/build-info.h.in")
        ret += call_subprocess("cp scripts/LlamaConfig.cmake.in out_root/scripts/LlamaConfig.cmake.in")
        if not ret:
            print("Error during copying files cmake script depends on:", test_config.command_output)

        ret += call_subprocess("cd out_root && git init && git add ./ && git commit -m \"raw migrated code\"")
        if not ret:
            print("Error during run git operation:", test_config.command_output)

        # Temporarily low the cmake minimum version required to 3.20.
        ret = call_subprocess("sed -i s/3.24/3.20/g ./out_root/CMakeLists.txt")
        if not ret:
            print("Error during replace cmake minimum version required:", test_config.command_output)

        if (os.path.exists("/opt/intel/oneapi/setvars.sh")):
            ret = call_subprocess("mkdir -p out_root/build && cd out_root/build && cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DLLAMA_CUBLAS=ON ../")
            if not ret:
                print("Error during cmake configure stage:", test_config.command_output)
            ret = call_subprocess("source /opt/intel/oneapi/setvars.sh --force && cd out_root/build && make")
            if not ret:
                print("Error during cmake build stage:", test_config.command_output)
        else:
            # For local machine test, "source /path/to/intel/oneapi/setvars.sh" is set in advance
            ret = call_subprocess("mkdir -p out_root/build && cd out_root/build && cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DLLAMA_CUBLAS=ON ../")
            if not ret:
                print("Error during cmake configure stage:", test_config.command_output)
            ret = call_subprocess("cd out_root/build && make")
            if not ret:
                print("Error during cmake build stage:", test_config.command_output)

        ret =  os.path.exists("out_root/build/bin/main")
        if not ret:
            print("llama target binary not exist:", test_config.command_output)
    else:
        print("Incorrect test name:", test_config.current_test)
    return ret

def run_test():
    return True
