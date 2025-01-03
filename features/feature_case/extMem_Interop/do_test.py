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

CMP_OPT = []
LNK_OPT = []

def setup_test():
    global CMP_OPT
    global LNK_OPT

    # Calculate vk_root based on the operating system
    # if INFO_RDRIVE is present in env (for CI)
    VK_ROOT = os.environ.get('INFO_RDRIVE')
    if VK_ROOT is not None:
        VK_ROOT = os.path.join(VK_ROOT, 'ref', 'VulkanSDK')

        VK_INC_DIR = ''
        VK_LINK_DIR = ''
        if (platform.system() == 'Windows'):
            VK_INC_DIR = os.path.join(VK_ROOT, 'include')
            VK_LINK_DIR = os.path.join(VK_ROOT, 'Lib')
        else:
            VK_INC_DIR = os.path.join(VK_ROOT, 'x86_64', 'include')
            VK_LINK_DIR = os.path.join(VK_ROOT, 'x86_64', 'lib')

        CMP_OPT.append('-I' + VK_INC_DIR)
        LNK_OPT.append('-L' + VK_LINK_DIR)

        LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH')
        os.environ['LD_LIBRARY_PATH'] = VK_LINK_DIR + ':' + LD_LIBRARY_PATH

    VK_LINK_OPT = 'vulkan'
    if (platform.system() == 'Windows'):
        VK_LINK_OPT += '-1'
    LNK_OPT.append('-l' + VK_LINK_OPT)

    return True

def migrate_test():
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), test_config.current_test)
    test_config.out_root = os.path.join(in_root, 'out_root')

    # Clean the out-root when it exisits.
    if os.path.exists(test_config.out_root):
        shutil.rmtree(test_config.out_root)
    
    src.append(os.path.abspath(os.path.join(in_root, 'extMem_interop_vk.cu')))
    src.append(' --use-experimental-features=bindless_images')

    extra_args.extend(CMP_OPT)

    return do_migrate(src, in_root, test_config.out_root, extra_args)

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)

    srcs = []
    srcs.append(os.path.abspath(os.path.join(test_config.out_root, 'extMem_interop_vk.dp.cpp')))

    return compile_and_link(srcs, cmpopts=CMP_OPT, linkopt=LNK_OPT)

def run_test():
    return call_subprocess(os.path.join(os.path.curdir, test_config.current_test + '.run'))
