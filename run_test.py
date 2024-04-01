# ====------ run_test.py---------- *- Python -* ----===##
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
import subprocess
from typing import DefaultDict
import xml.etree.ElementTree as ET
import os
import re
import platform
import sys
import importlib
import json
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
import multiprocessing as mp

from test_utils import *


class suite_config:
    def __init__(self, suite_name, test_config_map, suite_deps_files):
        self.name = suite_name
        self.test_config_map = test_config_map
        self.suite_deps_files = suite_deps_files
    def get_test_config_map(self):
        return self.test_config_map
    def get_suite_name(self):
        return self.name
    def get_suite_deps_files(self):
        return self.suite_deps_files

# OS only supported Windows and Linux
class platform_rule:
    def __init__(self, case_name, os_family, run_on_this_platform, cuda_verion="", cuda_range=""):
        self.name = case_name
        self.os_family = os_family
        self.run_on_this_platform = run_on_this_platform
        self.cuda_version = cuda_verion
        self.cuda_range = cuda_range

class option_rule:
    def __init__(self, case_name, exclude_option, only_option, not_double_type_feature):
        self.name = case_name
        self.exclude_option = exclude_option
        self.only_option = only_option
        self.not_double_type_feature = not_double_type_feature

class case_config:
    def __init__(self, case_name, test_dep_files, option_rule_list, platform_rule_list,
                split_group):
        self.name = case_name
        self.test_dep_files = test_dep_files
        self.option_rule_list = option_rule_list
        self.platform_rule_list = platform_rule_list
        self.split_group = split_group

def parse_suite_cfg(suite_name, root_path):
    xml_file = os.path.join(os.path.abspath(root_path), suite_name + ".xml")
    root = parse_xml_cfg_file(xml_file)
    suite_deps_files = []
    test_config_map = {}

    for test_case in root.iter("test"):
        case_name = str(test_case.get("testName"))
        case_config_path = str(test_case.get("configFile"))
        split_group = test_case.get("splitGroup")
        print_debug_log("case info: case name: ", case_name, case_config_path)
        case_cfg_obj = prepare_test_case(case_name, case_config_path, root_path, split_group)
        print_debug_log(case_name, case_cfg_obj.test_dep_files)
        test_config_map[case_name] = case_cfg_obj
    for file in root.iter("file"):
        suite_deps_files.append(file.get("path"))
    suite_cfg = suite_config(suite_name, test_config_map, suite_deps_files)
    return suite_cfg

def parse_xml_cfg_file(xml_path):
    if (not os.path.isfile(xml_path)):
        exit("The " + xml_path + " file is not exists. Please double check.")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root

def prepare_test_case(case_name, test_config_path, root_path, split_group):
    test_config_path = os.path.abspath(os.path.join(root_path, test_config_path))
    root = parse_xml_cfg_file(test_config_path)
    test_dirver = str(root.get("driverID"))
    case = ""
    if "name" in root.attrib:
        case = root.get("name")    # update the case name from the suite.xml
    else:
        case = case_name
    print_debug_log("test driver name: ",  test_dirver)
    print_debug_log("test case name: ", case)
    test_files = []
    option_rules = []
    platform_rule_list = []
    for file in root.iter("file"):
        file_path = str(file.get("path"))
        file_path = replace_test_name(case_name, file_path)
        if (not isabs(file_path)):
            file_path = os.path.join(os.path.abspath(root_path), file_path)
        if (os.path.exists(file_path)):
            test_files.append(file_path)
    for rule in root.iter("optlevelRule"):
        exclude_option = ""
        only_option = ""
        not_double_type_feature = ""
        if "excludeOptlevelNameString" in rule.attrib:
            exclude_option = str(rule.get("excludeOptlevelNameString"))
        if "onlyOptlevelNameString" in rule.attrib:
            only_option = str(rule.get("onlyOptlevelNameString"))
        if "GPUFeature" in rule.attrib:
            not_double_type_feature = str(rule.get("GPUFeature"))
        option_rules.append(option_rule(case_name, exclude_option, only_option, not_double_type_feature))

    for rule in root.iter("platformRule"):
        rule_osfamily = str(rule.get("OSFamily"))
        rule_test_on_this_plaform = str(rule.get("runOnThisPlatform"))
        rule_cuda_ver = ""
        rule_cuda_range = ""
        if "kit" in rule.attrib:
            rule_cuda_ver = str(rule.get("kit"))
            rule_cuda_range = str(rule.get("kitRange"))
        pr = platform_rule(case_name, rule_osfamily, rule_test_on_this_plaform, rule_cuda_ver, rule_cuda_range)
        platform_rule_list.append(pr)

    print_debug_log("test files ", test_files)
    print_debug_log("option rules are: ", option_rules)
    print_debug_log("platform rules: ", platform_rule_list)
    return case_config(case, test_files, option_rules, platform_rule_list, split_group)

def import_test_module(workspace):
    do_test_script = os.path.join(workspace, "do_test.py")
    driver_path = os.path.dirname(do_test_script)

    sys.path.append(driver_path)
    if os.path.exists(do_test_script):
        module = importlib.import_module("do_test")
        importlib.invalidate_caches()
        importlib.reload(module)
        return module
    return ""

def pop_module(workspace):
    do_test_script = os.path.join(workspace, "do_test.py")
    driver_path = os.path.dirname(do_test_script)
    if driver_path in sys.path:
        sys.path.pop()
    return ""

def test_setup(test_driver_module, specific_module):
    if not specific_module:
        return test_driver_module.setup_test()
    elif hasattr(specific_module, "setup_test"):
        return specific_module.setup_test()
    return True

def test_migrate(test_driver_module, specific_module):
    if not specific_module:
        return test_driver_module.migrate_test()
    elif hasattr(specific_module, "migrate_test"):
        return specific_module.migrate_test()
    return True

def test_build(test_driver_module, specific_module):
    if not specific_module:
        return test_driver_module.build_test()
    elif hasattr(specific_module, "build_test"):
        return specific_module.build_test()
    return True

def run_migrated_binary_test(test_driver_module, specific_module):
    if not specific_module:
        return test_driver_module.run_test()
    elif hasattr(specific_module, "run_test"):
        return specific_module.run_test()
    return True

# Execute the test driver to do the validation.
def run_test_driver(module):
    case_workspace = os.path.join(os.path.dirname(test_config.result_text), test_config.current_test)
    test_config.test_status = ""
    ret_val = True
    specific_module = ""

    if is_registered_module(case_workspace):
        specific_module = import_test_module(case_workspace)
    migrated = False
    built = False
    run = False

    ret_val = test_setup(module, specific_module)
    if ret_val:
        ret_val = test_migrate(module, specific_module)
        migrated = True
    else:
        test_config.test_status = "BADTEST"

    if ret_val:
        ret_val = test_build(module, specific_module)
        built = True
    elif migrated:
        test_config.test_status = "MIGFAIL"

    if ret_val:
        call_subprocess("sycl-ls")
        ret_val = run_migrated_binary_test(module, specific_module)
        run = True
    elif migrated and built:
        test_config.test_status = "COMPFAIL"

    if migrated and built and run and ret_val:
        test_config.test_status = "PASS"
    elif migrated and built and run:
        test_config.test_status = "RUNFAIL"

    if is_registered_module(case_workspace):
        pop_module(case_workspace)
    return ret_val

# To do: if the API was enabled in CUDA 9.2 version but deprecated in the CUDA 11.4 version,
# This change has not been covered yet. Currently, only cover the skip the older version run or
# skip the latest deprecated case running.
# Check whether the platform (CUDA header version and OS version) by rules.
# Rule1:skip < CUDA 9.2
# Rule2:skip > CUDA 11.4
# Not supported
def is_platform_supported(platform_rule_list):
    for platform_rule in platform_rule_list:
        if platform_rule.os_family != platform.system():
            continue
        if platform_rule.cuda_version:
            version = int(float(re.findall("\d+\.?\d*", platform_rule.cuda_version)[0])) * 1000
            print_debug_log("CUDA version is ", version)
            print_debug_log("default CUDA version is ", test_config.cuda_ver)
            print_debug_log("default CUDA range is ", platform_rule.cuda_range)
            if platform_rule.cuda_range == "LATER_OR_EQUAL" and test_config.cuda_ver >= version and platform_rule.run_on_this_platform.upper() == "FALSE":
                return False
            elif platform_rule.cuda_range == "OLDER" and test_config.cuda_ver < version and platform_rule.run_on_this_platform.upper() == "FALSE":
                return False
            elif platform_rule.cuda_range == "LATER" and test_config.cuda_ver > version and platform_rule.run_on_this_platform.upper() == "FALSE":
                return False
            elif platform_rule.cuda_range == "OLDER_OR_EQUAL" and test_config.cuda_ver <= version and platform_rule.run_on_this_platform.upper() == "FALSE":
                return False
            elif platform_rule.cuda_range == "EQUAL" and test_config.cuda_ver == version and platform_rule.run_on_this_platform.upper() == "FALSE":
                return False
        else:
            return platform_rule.run_on_this_platform.upper() == "TRUE"
    return True

def is_option_supported(option_rule_list):
    for option_rule in option_rule_list:
        if option_rule.exclude_option != "" and  option_rule.exclude_option in test_config.test_option and not option_rule.not_double_type_feature:
            return False
        elif option_rule.only_option not in test_config.test_option:
            return False
        elif option_rule.exclude_option in test_config.test_option and option_rule.not_double_type_feature == "NOT double":
            if test_config.backend_device not in test_config.support_double_gpu:
                return False
    return True

def test_single_case(current_test, single_case_config, workspace, suite_root_path):
    module = import_test_driver(suite_root_path)
    test_config.current_test = current_test
    test_config.executed_command = "================= " + test_config.current_test + " ==================\n"
    if single_case_config.platform_rule_list and not is_platform_supported(single_case_config.platform_rule_list):
        append_log_to_file(test_config.result_text, current_test + " Skip " + "\n")
        return True, current_test, "Skip"

    if single_case_config.option_rule_list and not is_option_supported(single_case_config.option_rule_list):
        append_log_to_file(test_config.result_text, current_test + " Skip " + "\n")
        return True, current_test, "Skip"

    case_workspace = os.path.join(workspace, current_test)
    if not os.path.exists(case_workspace):
        os.makedirs(case_workspace)
    os.chdir(workspace)

    test_config.log_file = os.path.join(workspace, current_test + ".lf")
    copy_source_to_ws(single_case_config.test_dep_files, case_workspace, suite_root_path)
    test_result = run_test_driver(module)
    # Write the execution command to command.tst, execution log to <test case>.lf, execution result to result.md
    write_log_to_file(test_config.log_file, "------------------------------------------------------------------------\n\n" + \
                "=================== "+ test_config.current_test + " is " + test_config.test_status + " ======================\n " + test_config.execution_log)
    append_log_to_file(test_config.result_text, test_config.current_test + " " + test_config.test_status + "\n")
    append_log_to_file(test_config.command_file, test_config.executed_command + "\n")

    if not test_result: # When execution failed, then print the failed log in the terminal.
        print_result(test_config.current_test, test_config.test_status, test_config.execution_log)
    return test_result, current_test, test_config.test_status

def prepare_test_workspace(root_path, suite_name, opt, case = ""):
    suite_workspace = os.path.join(os.path.abspath(root_path), suite_name, opt)
    case_workspace = os.path.join(suite_workspace, case)
    test_config.command_file = os.path.join(suite_workspace, "command.tst")
    test_config.result_text = os.path.join(suite_workspace, "result.md")

    if os.path.isdir(suite_workspace) and not case:
        shutil.rmtree(suite_workspace)
        os.makedirs(suite_workspace)
    elif not os.path.isdir(suite_workspace):
        os.makedirs(suite_workspace)
    return suite_workspace

# Split the GPU backend to double and none double kernel type.
def get_gpu_split_test_suite(suite_cfg):
    # Not specific the backend device, execute all the test cases.
    if not test_config.backend_device:
        return suite_cfg.test_config_map
    new_test_config_map = {}
    for current_test, case_config in suite_cfg.test_config_map.items():
        # Run the test case on the GPU device that support the double kernel type.
        if test_config.backend_device in test_config.support_double_gpu and case_config.split_group == "double":
            new_test_config_map[current_test] = case_config
        elif test_config.backend_device not in test_config.support_double_gpu and not case_config.split_group:
            new_test_config_map[current_test] = case_config
    return new_test_config_map

failed_cases = []
passed_cases = []
def collect_result(result):
    global failed_cases
    if not result[0]:
        failed_cases.append(result[1]) # Case Name
    else:
        passed_cases.append(result[1] + " " + result[2])
def test_suite(suite_root_path, suite_name, opt):
    test_ws_root = os.path.join(os.path.dirname(suite_root_path), "test_workspace")
    # module means the test driver for a test suite.
    module = import_test_driver(suite_root_path)
    test_config.suite_cfg = parse_suite_cfg(suite_name, suite_root_path)
    test_workspace = prepare_test_workspace(test_ws_root, suite_name, opt)
    cpu_count = os.cpu_count()
    pool = mp.Pool(cpu_count)
    test_config.suite_cfg.test_config_map = get_gpu_split_test_suite(test_config.suite_cfg)
    for current_test, single_case_config in test_config.suite_cfg.test_config_map.items():
        args = (current_test, single_case_config, test_workspace, suite_root_path)
        pool.apply_async(test_single_case, args, callback=collect_result)
    pool.close()
    pool.join()
    # Second round: single thread run the failed cases.
    test_config.is_first_round = False
    global failed_cases
    global passed_cases
    failed_logs = []
    for current_test in failed_cases:
        ret_val, test_name, test_status = test_single_case(current_test, test_config.suite_cfg.test_config_map[current_test], test_workspace, suite_root_path)
        if not ret_val:
            failed_logs.append(test_name + " " + test_status)
        else:
            passed_cases.append(test_name + " " + test_status)
    if passed_cases:
        print("===============passed case(s) ==========================")
        for case_log in passed_cases:
            print(case_log + " \n")
        print("========================================================")
    if failed_logs:
        print("===============Failed case(s) ==========================")
        for case_log in failed_logs:
            print(case_log + " \n")
        print("========================================================")
        return False
    return True

def test_single_case_in_suite(suite_root_path, suite_name, case, option):
    test_config.is_first_round = False # Disable the multi-thread effect to set the is_first_round to False
    test_ws_root = os.path.join(os.path.dirname(suite_root_path), "test_workspace")
    test_config.suite_cfg = parse_suite_cfg(suite_name, suite_root_path)
    test_workspace = prepare_test_workspace(test_ws_root, suite_name, option, case)

    config_running_device(option)
    if case not in test_config.suite_cfg.test_config_map.keys():
        exit("The test case " + case + " is not in the " + suite_name + " test suite! Please double check.")
    single_case_config = test_config.suite_cfg.test_config_map[case]
    retval, name, status = test_single_case(case, single_case_config, test_workspace, suite_root_path)
    return retval

# Before run the test:
# 1. Please specify the CUDA header files with CUDA_INCLUDE_PATH environment variable.
# 2. Please specify oneMKL build and run environment by following oneAPI document (some test cases do depend on oneMKL).
def check_deps():
    if not os.getenv("CUDA_INCLUDE_PATH") or not os.path.exists(os.environ["CUDA_INCLUDE_PATH"]):
        exit("Please setup CUDA_INCLUDE_PATH environment variable to specify the CUDA header file.")
    test_config.include_path = os.environ["CUDA_INCLUDE_PATH"]
    test_config.cuda_ver = get_cuda_version()
    if not test_config.cuda_ver:
        exit("The CUDA header files have not been detected. Please double check setting for CUDA header files.")

    # Please specify oneMKL build and run environment by following oneAPI document.
    if os.getenv("MKLROOT"):
        mkl_header = os.path.join(os.environ["MKLROOT"], "include", "mkl.h")
        if not os.path.exists(mkl_header) and os.getenv("MKLVER"):
            mkl_header = os.path.join(os.environ["MKLROOT"], os.environ["MKLVER"], "include", "mkl.h")
    else:
        exit("MKL header files have not been found. Please check setting for oneMKL.")

def do_sanity_test():
    check_deps()

def import_test_driver(suite_folder):
    sys.path.insert(0, suite_folder)
    test_driver_path = os.path.join(suite_folder, "test_drivers.xml")
    if os.path.exists(test_driver_path):
        root = parse_xml_cfg_file(test_driver_path)
        for test_driver in root.iter("testDriver"):
            test_config.test_driver = str(test_driver.get("driverID"))
    return importlib.import_module(test_config.test_driver)

def clean_global_setting():
    test_config.current_test = ""
    test_config.command_file = ""  # Used to store the executed command.
    test_config.log_file = ""      # Used to store the executed log for each case.
    test_config.result_text = ""   # Used to store the executed status for each case.
    test_config.out_root = ""
    test_config.subprocess_stdout_log = ""
    test_config.test_status = ""   # Default: "SKIPPED"
    test_config.test_driver = ""
    test_config.executed_command = ""
    test_config.execution_log = ""

# Parse the test suite configuration file and get the supported suite list.
def get_suite_list():
    xml_file = test_config.suite_list_file
    root = parse_xml_cfg_file(xml_file)
    suite_list = {}
    for suite in root.iter("suite"):
        suite_cfg = []
        suite_name = suite.get("name")
        suite_cfg.append(suite.get("dir"))
        suite_cfg.append(suite.get("opts"))
        suite_list[suite_name] = suite_cfg
    return suite_list

def config_running_device(opt):
    if "cpu" in opt:
        test_config.device_filter = "opencl:cpu"
    if "gpu" in opt:
        test_config.device_filter = "level_zero:gpu"
    if "cuda" in opt:
        test_config.device_filter = "cuda:gpu"
    test_config.migrate_option = test_config.option_map[opt]

def test_suite_with_opt(suite_root_path, suite_name, opt):
    config_running_device(opt)
    return test_suite(suite_root_path, suite_name, opt)

# Test the suite with the opts list.
def test_suite_with_opts(suite_root_path, suite_name, opts):
    ret = True
    for opt in opts:
        os.chdir(test_config.root_path)
        ret = test_suite_with_opt(suite_root_path, suite_name, opt.strip()) and ret
    return ret

def test_full_suite_list(suite_list):
    ret = True
    for suite_name in suite_list:
        clean_global_setting()
        os.chdir(test_config.root_path)
        suite_root_path = suite_list[suite_name][0]
        if os.path.exists(suite_root_path):
            suite_root_path = os.path.abspath(suite_root_path)
        opts = suite_list[suite_name][1].split(",")
        ret = test_suite_with_opts(suite_root_path, suite_name, opts) and ret
    return ret

def get_option_mapping():
    with open("option_mapping.json") as f:
        return json.load(f)

def define_global_test_option(opt, option_list):
    if opt not in option_list:
        sys.stderr.write("Must specify the option for test_suite_list.xml")
        exit(1)
    test_config.test_option = opt
    return True

def parse_input_args():
    parser = argparse.ArgumentParser(description = "Test driver for C2S tool. \n", formatter_class=RawTextHelpFormatter,
             epilog="Examples: \n"  +
             "Run the single test case on the CPU device with default dpct option:\n\n" +
             "  python3 run_test.py --suite <test suite> --case <test case> --option option_cpu \n\n" +
             "Run the single test suite on the CPU device with default dpct option:\n\n" +
             "  python3 run_test.py --suite <test suite> --option <option_cpu> \n\n" +
             "Run the full test suites in the test_suite_list.xml: \n\n" +
             "  python3 run_test.py")
    parser.add_argument("--suite", "-s",  action = "store", default = "",
                    help = "The suites to run. e.g.: -s regressions. Please ref the test_suite_list.xml ")
    parser.add_argument("--case", "-c", action = "store", default = "",
                    help = "The test of suite to run. e.g.: -c simple_add. Please ref the <suite>.xml")
    parser.add_argument("--option", "-o", action = "store", default = "",
                    help = "The option applies to test. e.g.: -o option_cpu. Please ref the option_mapping.json")
    parser.add_argument("--device", '-d', action = "store", default = "",
                    help = "Current support Gen9, Gen12 and PVC backend device. e.g.: -d Gen9")
    args = parser.parse_args()
    if args.option and not args.suite:
        sys.stderr.write("Must specify the suite target to run.\n")
        exit(1)
    if args.suite and not args.option:
        sys.stderr.write("Must specify the option for suite target to run.\n")
        exit(1)
    if args.case and not args.suite:
        sys.stderr.write("Must specify the suite target to run.\n")
        exit(1)
    if args.device and args.device not in test_config.gpu_device:
        sys.stderr.write("Only support Gen9, Gen12 and PVC GPU device.\n")
        exit(1)
    return args

def main():
    args = parse_input_args()
    do_sanity_test()
    set_default_compiler(args.option == 'option_cuda_backend')
    set_default_c_compiler(args.option == 'option_cuda_backend')
    suite_list = get_suite_list()
    test_config.root_path = os.getcwd()
    test_config.option_map = get_option_mapping()
    test_config.backend_device = args.device

    ret = True
    # Run all the tests in the test_suite_list.xml
    if not args.suite:
        ret = test_full_suite_list(suite_list)
    else:
        suite_root_path = os.path.abspath(suite_list[args.suite][0])
        define_global_test_option(args.option, suite_list[args.suite][1])
        if not args.case:
            ret = test_suite_with_opt(suite_root_path, args.suite, args.option)
        elif args.case:
            ret = test_single_case_in_suite(suite_root_path, args.suite, args.case, args.option)
    if not ret:
        sys.stderr.write("Some test case(s) fail\n")
        exit(-1)
    print("----------------Test pass-----------------")

if __name__ == "__main__":
    main()
