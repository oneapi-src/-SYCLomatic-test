# ====------ test_feature.py---------- *- Python -* ----===##
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
from pathlib import Path
import fileinput
import shutil

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

exec_tests = ['asm', 'asm_bar', 'asm_mem', 'asm_atom', 'asm_arith', 'asm_vinst', 'asm_v2inst', 'asm_v4inst', 'asm_optimize', 'thrust-vector-2', 'thrust-binary-search', 'thrust-count', 'thrust-copy',
              'thrust-qmc', 'thrust-transform-if', 'thrust-policy', 'thrust-list', 'module-kernel',
              'kernel-launch', 'thrust-gather', 'thrust-gather_if', 'cub_device_partition',
              'thrust-scatter', 'thrust-unique_by_key_copy', 'thrust-for-hypre', 'thrust-merge_by_key',
              'thrust-rawptr-noneusm', 'driverStreamAndEvent', 'grid_sync', 'deviceProp', 'gridThreads', 'kernel_library', 'cub_block_p2', 'cub_device_spmv',
              'cub_constant_iterator', 'cub_device_reduce_max', 'cub_device_reduce_min', 'cub_discard_iterator', 'ccl-test', 'ccl-test2', 'ccl-test3', 'ccl-error',
              'cub_device', 'cub_device_reduce_sum', 'cub_device_reduce', 'cub_device_reduce_by_key', 'cub_device_select_unique_by_key', 'cub_device_segmented_sort_keys',
              'cub_device_scan_inclusive_scan', 'cub_device_scan_exclusive_scan', 'cub_device_seg_radix_sort_pairs', 'cub_device_no_trivial_runs', 'cub_device_merge_sort.cu',
              'cub_device_scan_inclusive_sum', 'cub_device_scan_exclusive_sum', 'cub_device_select_unique', 'cub_device_radix_sort_keys', 'cub_device_radix_sort_pairs',
              'cub_device_select_flagged', 'cub_device_run_length_encide_encode', 'cub_counting_iterator', 'cub_arg_index_input_iterator', 'cub_device_seg_radix_sort_keys',
              'cub_device_inclusive_sum_by_key', 'cub_device_exclusive_sum_by_key', 'cub_device_inclusive_scan_by_key', 'cub_device_exclusive_scan_by_key',
              'cub_device_reduce_arg', 'cub_device_seg_sort_pairs', 'cub_intrinsic', 'cub_device_seg_sort_keys', 'thrust-math1', 'thrust-math2', 'cub_block_exchange',
              'cub_transform_iterator', 'activemask', 'complex', 'thrust-math', 'libcu_array', 'libcu_complex', 'libcu_tuple', 'cub_block_radix_sort',
              'user_defined_rules', 'math-exec', 'math-intrinsics', 'math-habs', 'math-emu-double', 'math-emu-float', 'math-emu-half', 'math-emu-half-after11', 'math-emu-half2', 'math-emu-half2-after11', 'math-emu-half2-after12', 'math-emu-simd',
              'math-emu-bf16', 'math-emu-bf162-after12', 'math-emu-bf162', 'math-experimental-bf16', 'math-experimental-bf162', "math-half-raw",
              'math-ext-bf16-conv', 'math-ext-double', 'math-ext-float', 'math-ext-half', 'math-ext-half-after11', 'math-ext-half-conv', 'math-ext-half2', 'math-ext-half2-after11', 'math-ext-simd', 'cudnn-activation',
              'cudnn-fill', 'cudnn-lrn', 'cudnn-memory', 'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax',
              'cudnn-sum', 'math-funnelshift', 'thrust-sort_by_key', 'thrust-find', 'thrust-inner_product', 'thrust-reduce_by_key',
              'math-bf16-conv', 'math-emu-bf16-conv-double', 'math-ext-bf16-conv-double', 'math-half-conv', 'math-int',
              'math-bfloat16', 'libcu_atomic', 'test_shared_memory', 'cudnn-reduction', 'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3',
              'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3', 'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5', 'cudnn-convp6', 'cudnn-convp7', 
              'cudnn_mutilple_files', "cusparse_1", "cusparse_2", "cusparse_3", "cusparse_4", "cusparse_5", "cusparse_6", "cusparse_7", "cusparse_8", "cusparse_9", "cusparse_10",
              'cudnn-GetErrorString', 'cub_device_histgram', 'peer_access', 'driver_err_handle',
              'cudnn-types', 'cudnn-version', 'cudnn-dropout', 'const_opt',
              'constant_attr', 'sync_warp_p2', 'occupancy_calculation',
              'text_experimental_obj_array', 'text_experimental_obj_driver_api', 'text_experimental_obj_mipmap', 'text_experimental_obj_linear', 'text_experimental_obj_pitch2d',
              'text_obj_array', 'text_obj_linear', 'text_obj_pitch2d', 'match',
              'curand-device2', 'curandEnum', 'codepin_all_public_dump',
              'thrust-unique_by_key', 'cufft_test', 'cufft-external-workspace', "pointer_attributes", 'math_intel_specific', 'math-drcp', 'thrust-pinned-allocator', 'driverMem',
              'cusolver_test1', 'cusolver_test2', 'cusolver_test3', 'cusolver_test4', 'cusolver_test5', 'thrust_op', 'cublas-extension', 'cublas_v1_runable', 'thrust_minmax_element',
              'thrust_is_sorted', 'thrust_partition', 'thrust_remove_copy', 'thrust_unique_copy', 'thrust_transform_exclusive_scan',
              'thrust_set_difference', 'thrust_set_difference_by_key', 'thrust_set_intersection_by_key', 'thrust_stable_sort',
              'thrust_tabulate', 'thrust_for_each_n', 'device_info', 'defaultStream', 'cudnn-rnn', 'feature_profiling',
              'thrust_raw_reference_cast', 'thrust_partition_copy', 'thrust_stable_partition_copy', 'device_global',
              'thrust_stable_partition', 'thrust_remove', 'cub_device_segmented_sort_pairs', 'thrust_find_if_not',
              'thrust_find_if', 'thrust_mismatch', 'thrust_replace_copy', 'thrust_reverse', 'cooperative_groups_reduce', 'cooperative_groups_thread_group', 'cooperative_groups_data_manipulate',
              'remove_unnecessary_wait', 'thrust_equal_range', 'thrust_transform_inclusive_scan', 'thrust_uninitialized_copy_n', 'thrust_uninitialized_copy',
              'thrust_random_type', 'thrust_scatter_if', 'thrust_all_of', 'thrust_none_of', 'thrust_is_partitioned',
              'thrust_is_sorted_until', 'thrust_set_intersection', 'thrust_set_union_by_key', 'thrust_set_union',
              'thrust_swap_ranges', 'thrust_uninitialized_fill_n', 'thrust_equal', 'system_atomic', 'thrust_detail_types',
              'operator_eq', 'operator_neq', 'operator_lege', 'thrust_system', 'thrust_reverse_copy',
              'thrust_device_new_delete', 'thrust_temporary_buffer', 'thrust_malloc_free', 'codepin', 'thrust_unique_count',
              'thrust_advance_trans_op_itr', 'cuda_stream_query', "matmul", "transform"]

occupancy_calculation_exper = ['occupancy_calculation']

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
    if test_config.current_test == 'cufft_test':
        return do_migrate([os.path.join(in_root, 'cufft_test.cu')], in_root, test_config.out_root, extra_args)

    for dirpath, dirnames, filenames in os.walk(in_root):
        for filename in [f for f in filenames if re.match('.*(cu|cpp|c)$', f)]:
            src.append(os.path.abspath(os.path.join(dirpath, filename)))

    nd_range_bar_exper = ['grid_sync']
    logical_group_exper = ['cooperative_groups', 'cooperative_groups_thread_group', 'cooperative_groups_data_manipulate']
    uniform_group_exper = ['cooperative_group_coalesced_group']
    experimental_bfloat16_tests = ['math-experimental-bf16', 'math-experimental-bf162']

    if test_config.current_test in nd_range_bar_exper:
        src.append(' --use-experimental-features=nd_range_barrier ')
    if test_config.current_test == "user_defined_rules":
        src.append(' --rule-file=./user_defined_rules/rules.yaml')
    if test_config.current_test in logical_group_exper:
        src.append(' --use-experimental-features=logical-group ')
    if test_config.current_test in uniform_group_exper:
        src.append(' --use-experimental-features=non-uniform-groups ')
    if test_config.current_test == 'math_intel_specific':
        src.append(' --rule-file=./math_intel_specific/intel_specific_math.yaml')
    if test_config.current_test.startswith('math-ext-'):
        src.append(' --use-dpcpp-extensions=intel_device_math')
    if test_config.current_test in occupancy_calculation_exper:
        src.append(' --use-experimental-features=occupancy-calculation ')
    if test_config.current_test == 'feature_profiling':
        src.append(' --enable-profiling ')
    if test_config.current_test == 'asm_bar':
        src.append(' --use-experimental-features=non-uniform-groups ')
    if test_config.current_test == 'cub_block':
        src.append(' --use-experimental-features=user-defined-reductions ')
    if test_config.current_test == 'device_global':
        src.append(' --use-experimental-features=device_global ')
    if test_config.current_test == 'sync_warp_p2':
        src.append(' --use-experimental-features=masked-sub-group-operation ')
    if test_config.current_test == 'wmma' or test_config.current_test == 'wmma_type':
        src.append(' --use-experimental-features=matrix ')
    if test_config.current_test in experimental_bfloat16_tests:
        src.append(' --use-experimental-features=bfloat16_math_functions ')
    if test_config.current_test == 'const_opt' or test_config.current_test == 'asm_optimize':
        src.append(' --optimize-migration ')
    if test_config.current_test.startswith('text_experimental_'):
        src.append(' --use-experimental-features=bindless_images')
    if "codepin" in test_config.current_test:
        src.append(' --enable-codepin ')
    return do_migrate(src, in_root, test_config.out_root, extra_args)

def manual_fix_for_cufft_external_workspace(migrated_file):
    lines = []
    is_first_occur = True
    with open(migrated_file) as in_f:
        for line in in_f:
            if ('&workSize' in line):
                if (is_first_occur):
                    line = line.replace('&workSize', '&workSize, std::pair(dpct::fft::fft_direction::forward, true)')
                    is_first_occur = False
                else:
                    line = line.replace('&workSize', '&workSize, std::pair(dpct::fft::fft_direction::backward, true)')
            lines.append(line)
    with open(migrated_file, 'w') as out_f:
        for line in lines:
            out_f.write(line)

def manual_fix_for_occupancy_calculation(migrated_file):
    lines = []
    with open(migrated_file) as in_f:
        for line in in_f:
            if ('dpct_placeholder' in line):
                line = line.replace('dpct_placeholder', '0')
            lines.append(line)
    with open(migrated_file, 'w') as out_f:
        for line in lines:
            out_f.write(line)

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_options = []
    link_opts = []
    objects = []

    oneDPL_related = ['thrust-vector', 'thrust-for-h2o4gpu', 'thrust-for-RapidCFD', 'cub_device',
             'cub_block_p2']

    oneDNN_related = ['cudnn-activation', 'cudnn-fill', 'cudnn-lrn', 'cudnn-memory',
             'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax', 'cudnn-sum', 'cudnn-reduction',
             'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3', 'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3',
             'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5', 'cudnn-convp6', 'cudnn-rnn',
             'cudnn-GetErrorString', 'cudnn-convp7',
             'cudnn-types', 'cudnn-version', 'cudnn-dropout', 'matmul'
             ]

    no_fast_math_tests = ['math-emu-half-after11', 'math-emu-half2-after11', 'math-ext-half-after11', 'math-ext-half2-after11',
                          'math-emu-bf16', 'math-emu-bf162', 'math-experimental-bf16', 'math-experimental-bf162']

    if test_config.current_test in oneDPL_related:
        cmp_options.append(prepare_oneDPL_specific_macro())

    if test_config.current_test == 'cub_device_spmv' or re.match('^cu.*', test_config.current_test):
        if platform.system() == 'Linux':
            link_opts = test_config.mkl_link_opt_lin
        else:
            link_opts = test_config.mkl_link_opt_win
        cmp_options.append("-DMKL_ILP64")

    if test_config.current_test in no_fast_math_tests:
        cmp_options.append("-fno-fast-math")

    if test_config.current_test.startswith('ccl-'):
        link_opts.append('-lccl -lmpi')

    if "codepin" in test_config.current_test:
        test_config.out_root = test_config.out_root + "_codepin_sycl"

    if test_config.current_test == 'device_global':
        if platform.system() == 'Linux':
            cmp_options.append("-std=c++20")
        else:
            cmp_options.append("-Qstd=c++20")

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        for filename in [f for f in filenames if re.match('.*(cpp|c)$', f)]:
            srcs.append(os.path.abspath(os.path.join(dirpath, filename)))
    if platform.system() == 'Linux':
        link_opts.append(' -lpthread ')
    if test_config.current_test in oneDNN_related:
        if platform.system() == 'Linux':
            link_opts.append(' -ldnnl')
        else:
            link_opts.append(' dnnl.lib')
    ret = False

    if (test_config.current_test == 'cufft-external-workspace'):
        manual_fix_for_cufft_external_workspace(srcs[0])
    if (test_config.current_test in occupancy_calculation_exper):
        manual_fix_for_occupancy_calculation(srcs[0])

    if test_config.current_test == 'cufft_test':
        ret = compile_and_link([os.path.join(test_config.out_root, 'cufft_test.dp.cpp')], cmp_options, objects, link_opts)
    elif test_config.current_test in exec_tests:
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    elif re.match('^cufft.*', test_config.current_test) and platform.system() == 'Linux':
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    else:
        ret = compile_files(srcs, cmp_options)
    return ret


def run_test():
    if test_config.current_test not in exec_tests:
        return True
    if test_config.current_test.startswith('text_experimental_obj_') and test_config.device_filter.count("cuda") == 0:
        return True
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    os.environ['CL_CONFIG_CPU_EXPERIMENTAL_FP16']="1"
    if test_config.current_test.startswith('ccl-test'):
        return call_subprocess('mpirun -n 2 ' + os.path.join(os.path.curdir, test_config.current_test + '.run '))
    return run_binary_with_args()

