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

options = [("in-root", "Attribute", ["Analysis", "Migration", "BuildScript"], "a"),
("out-root", "Attribute", ["Migration", "BuildScript"], "a"),
("cuda-include-path", "Attribute", ["Analysis", "Query", "Migration", "BuildScript"], "a"),
("report-file-prefix", "Attribute", ["Migration"], "a"),
("report-only", "Attribute", ["Migration"], ""),
("keep-original-code", "Attribute", ["Migration"], ""),
("report-type", "Attribute", ["Migration"], "all"),
("report-format", "Attribute", ["Migration"], "csv"),
("suppress-warnings", "Attribute", ["Migration"], "a"),
("suppress-warnings-all", "Attribute", ["Migration"], ""),
("stop-on-parse-err", "Attribute", ["Migration", "Analysis"], ""),
("check-unicode-security", "Attribute", ["Migration", "Analysis"], ""),
("enable-profiling", "Attribute", ["Migration", "Analysis"], ""),
("sycl-named-lambda", "Attribute", ["Migration", "Analysis"], ""),
("output-verbosity", "Attribute", ["Migration"], "detailed"),
("output-file", "Attribute", ["Migration"], "a"),
("rule-file", "Attribute", ["Migration", "BuildScript", "Analysis"], "a"),
("usm-level", "Attribute", ["Migration", "Analysis"], "none"),
("migrate-build-script", "Attribute", ["Migration", "Analysis"], "CMake"),
("format-range", "Attribute", ["Migration"], "all"),
("format-style", "Attribute", ["Migration"], "custom"),
("no-dry-pattern", "Attribute", ["Migration", "Analysis"], ""),
("process-all", "Attribute", ["Migration", "Analysis"], ""),
("enable-codepin", "Attribute", ["Migration"], ""),
("enable-ctad", "Attribute", ["Migration", "Analysis"], ""),
("comments", "Attribute", ["Migration"], ""),
("always-use-async-handler", "Attribute", ["Migration", "Analysis"], ""),
("assume-nd-range-dim", "Attribute", ["Migration", "Analysis"], "1"),
("use-explicit-namespace", "Attribute", ["Migration", "Analysis"], "dpct"),
("no-dpcpp-extensions", "Attribute", ["Migration", "Analysis"], "bfloat16"),
("use-dpcpp-extensions", "Attribute", ["Migration", "Analysis"], "c_cxx_standard_library"),
("use-experimental-features", "Attribute", ["Migration", "Analysis"], "free-function-queries"),
("gen-build-script", "Attribute", ["Migration", "Analysis"], ""),
("build-script-file", "Attribute", ["Migration"], "a"),
("migrate-build-script-only", "Action", ["BuildScript"], ""),
("in-root-exclude", "Attribute", ["BuildScript", "Migration", "Analysis"], "a"),
("optimize-migration", "Attribute", ["Analysis", "Migration"], ""),
("no-incremental-migration", "Attribute", ["Analysis", "Migration"], ""),
("analysis-scope-path", "Attribute", ["Analysis", "Migration"], "a"),
("change-cuda-files-extension-only", "Attribute", ["Migration"], ""),
("sycl-file-extension", "Attribute", ["Migration"], "dp-cpp"),
("gen-helper-function", "Attribute", ["Migration"], ""),
("helper-function-dir", "Action", ["Help"], ""),
("query-api-mapping", "Action", ["Query"], "a"),
("helper-function-preference", "Attribute", ["Analysis", "Migration"], "no-queue-device"),
("analysis-mode", "Action", ["Analysis"], ""),
("analysis-mode-output-file", "Attribute", ["Analysis"], "a")]

def add_option(option):
    ret = " -" + option[0]
    if len(option[3]):
      ret += '=' + option[3]
    return ret

def add_options(options = list):
    ret = ""
    for opt in options:
        ret += add_option(opt)
    return ret

def check_result(cmd, action, attributes):
    call_subprocess(test_config.CT_TOOL + cmd)
    expect = "Warning: Option \"-{0}\" is ignored because it conflicts with option \"-" + action[0] + "\"."
    for attr in attributes:
        if action[2][0] in attr[2]:
            continue
        if not is_sub_string(expect.format(attr[0]), test_config.command_output):
            print("\"" + expect.format(attr[0]) + "\"not found")
            return False
    return True

def check_action(action, attributes):
    print("Check option conflict for " + action[0])
    attr_opts = add_options(attributes)
    return check_result(add_option(action) + attr_opts, action, attributes) and check_result(attr_opts + add_option(action), action, attributes)

def check_confict(act1, act2, attribute_options):
    print("Check option conflict between " + act1[0] + " and " + act2[0])
    cmd = test_config.CT_TOOL
    cmd += add_option(act1)
    cmd += attribute_options
    cmd += add_option(act2)
    call_subprocess(cmd)
    expect = "Error: Option \"-{0}\" and option \"-{1}\" can not be used together."
    return is_sub_string(expect.format(act2[0], act1[0]), test_config.command_output)

def check_action_confilcts(actions, attributes):
    attr_opts = add_options(attributes)
    for i in range(len(actions)):
        for j in range(len(actions)):
            if i==j:
                continue
            if not check_confict(actions[i], actions[j], attr_opts):
                return False
    return True

def check_dependency(dependee, depender):
    print("Check dependency: " + depender + " depends on " + dependee)
    for opt in options:
        if depender == opt[0]:
            depender_opt = opt

    cmd = test_config.CT_TOOL
    cmd += add_option(depender_opt)
    call_subprocess(cmd)
    expect = "Error: Option \"-{0}\" require(s) that option \"-{1}\" be specified explicitly."
    if not is_sub_string(expect.format(depender, dependee), test_config.command_output):
        print("\"" + expect.format(depender, dependee) + "\" not found")
        return False

    return True

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    actions = []
    attributes = []
    for opt in options:
        if opt[1] == "Action":
            actions.append(opt)
        elif opt[1] == "Attribute":
            attributes.append(opt)

    for act in actions:
        if not check_action(act, attributes):
            return False

    if not check_action_confilcts(actions, attributes):
        return False

    if not check_dependency("gen-build-script", "build-script-file"):
        return False

    if not check_dependency("in-root", "process-all"):
        return False

    return True

def build_test():
    return True

def run_test():
    return True
