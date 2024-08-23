# ===------- do_test.py--------------------------------- *- Python -* -----===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    res = True

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--gen-build")
    reference = '--gen-build-script\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-gen-build")
    reference = '-gen-build-script\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=foo")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--output-verbosity=#d")
    reference = 'detailed\n' + \
                'diagnostics\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-output-verbosity=#d")
    reference = 'detailed\n' + \
                'diagnostics\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--output-verbosity=")
    reference = 'detailed\n' + \
                'diagnostics\n' + \
                'normal\n' + \
                'silent\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-output-verbosity=")
    reference = 'detailed\n' + \
                'diagnostics\n' + \
                'normal\n' + \
                'silent\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=foo#bar##--enable-ct")
    reference = '--enable-ctad\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=foo#bar##--enable-code")
    reference = '--enable-codepin\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=foo#bar###--format-range=#a")
    reference = 'all\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--rule-file=")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--rule-file")
    reference = '--rule-file\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-p=")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-p")
    reference = '-p\n' + \
                '-process-all\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--usm-level=#none,restricted#--use-explicit-namespace=#sycl,")
    reference = 'sycl,dpct\n' + \
                'sycl,none\n' + \
                'sycl,sycl\n' + \
                'sycl,sycl-math\n' + \
                'sycl,syclcompat\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=--usm-level=#none,restricted#--use-explicit-namespace=#sycl,s")
    reference = 'sycl,sycl\n' + \
                'sycl,sycl-math\n' + \
                'sycl,syclcompat\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=,")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete==")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=,,")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=-")
    opts = ['--always-use-async-handler\n',
            '--analysis-scope-path\n',
            '--assume-nd-range-dim=\n',
            '--build-script-file\n',
            '--check-unicode-security\n',
            '--comments\n',
            '--cuda-include-path\n',
            '--enable-ctad\n',
            '--enable-profiling\n',
            '--extra-arg\n',
            '--format-range=\n',
            '--format-style=\n',
            '--gen-build-script\n',
            '--help=\n',
            '--in-root\n',
            '--in-root-exclude\n',
            '--keep-original-code\n',
            '--no-dpcpp-extensions=\n',
            '--no-dry-pattern\n',
            '--no-incremental-migration\n',
            '--optimize-migration\n',
            '--out-root\n',
            '--output-file\n',
            '--output-verbosity=\n',
            '--process-all\n',
            '--report-file-prefix\n',
            '--report-format=\n',
            '--report-only\n',
            '--report-type=\n',
            '--rule-file\n',
            '--stop-on-parse-err\n',
            '--suppress-warnings\n',
            '--suppress-warnings-all\n',
            '--sycl-named-lambda\n',
            '--use-dpcpp-extensions=\n',
            '--use-experimental-features=\n',
            '--use-explicit-namespace=\n',
            '--usm-level=\n',
            '--version\n',
            '-p\n']
    for opt in opts:
        res = res and (opt in test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=##")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    call_subprocess(test_config.CT_TOOL + " --autocomplete=#")
    reference = '\n'
    res = res and (reference == test_config.command_output)

    return res

def build_test():
    return True

def run_test():
    return True
