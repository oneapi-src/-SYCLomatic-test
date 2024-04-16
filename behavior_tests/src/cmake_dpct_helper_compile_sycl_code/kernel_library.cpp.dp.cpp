// ====-------------- kernel_library.cpp.dp.cpp---------- -*- C++ -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "shared.hpp"
#include <dpct/dpct.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>

#define PTXFILE "premade.ptx"

static void loadModule(dpct::kernel_library &module, const char *ptxfile,
                       int load_type) {
  if (load_type == 0) {
    /*
    DPCT1103:0: 'ptxfile' should be a dynamic library. The dynamic library
    should supply wrapped kernel functions.
    */
    checkErrors(DPCT_CHECK_ERROR(
        DPCT_CHECK_ERROR(module = dpct::load_kernel_library(ptxfile))));
  } else if (load_type == 1 || load_type == 2) {
    std::ifstream ifile(ptxfile, std::ios::in | std::ios::binary);
    std::stringstream strStream;
    std::string content;

    // put file into a buffer for loading
    strStream << ifile.rdbuf();
    content = strStream.str();

    if (load_type == 1) {
      /*
      DPCT1104:1: 'content.c_str()' should point to a dynamic library loaded in
      memory. The dynamic library should supply wrapped kernel functions.
      */
      checkErrors(DPCT_CHECK_ERROR(DPCT_CHECK_ERROR(
          module = dpct::load_kernel_library_mem(content.c_str()))));
    } else {
#define LOADDATAEX_OPTIONS
#ifdef LOADDATAEX_OPTIONS
      const unsigned int jitNumOptions = 1;
      int jitOptions[jitNumOptions];
      void *jitOptVals[jitNumOptions];

      // set up wall clock time
      /*
      DPCT1048:2: The original value CU_JIT_WALL_TIME is not meaningful in the
      migrated code and was removed or replaced with 0. You may need to check
      the migrated code.
      */
      jitOptions[0] = 0;
#endif

#ifdef LOADDATAEX_OPTIONS
      /*
      DPCT1104:3: 'content.c_str()' should point to a dynamic library loaded in
      memory. The dynamic library should supply wrapped kernel functions.
      */
      checkErrors(DPCT_CHECK_ERROR(DPCT_CHECK_ERROR(
          module = dpct::load_kernel_library_mem(content.c_str()))));
#else
      checkErrors(cuModuleLoadDataEx(&module, content.c_str(), 0, NULL, NULL));
#endif

#ifdef LOADDATAEX_OPTIONS
      unsigned long long opt_int = (unsigned long long)jitOptVals[0];
      float jitTime = *((float *)&opt_int);

      std::cout << "\tcompile-time: " << jitTime << "ms\n";
#endif
    }
  }
}

int run_tests() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  static int cnt;
  int load_type;
  int invoke_type;

  for (load_type = 0; load_type < 3; load_type++) {
    for (invoke_type = 0; invoke_type < 2; invoke_type++) {
      int seed = SEED;

      // allocate memory
      int *a;
      int *b;
      int *c;

      a = sycl::malloc_shared<int>(VEC_LENGTH, q_ct1);
      b = sycl::malloc_shared<int>(VEC_LENGTH, q_ct1);
      c = sycl::malloc_shared<int>(VEC_LENGTH, q_ct1);

      for (int i = 0; i < VEC_LENGTH; i++) {
        b[i] = i + load_type * 53 + invoke_type * 97 + cnt * 101;
        c[i] = i + 1 + load_type * 53 + invoke_type * 97 + cnt * 101;
      }

      // test ptx module-loading, kernel function invocation, and verify results

      // load_type==0 use cuModuleLoad
      // load_type==1 use cuModuleLoadData
      // load_type==2 use cuModuleLoadDataEx

      // load pre-made ptx
      dpct::kernel_library module;
      loadModule(module, PTXFILE, load_type);

      if (module == NULL) {
        std::cout << "No module\n";
        exit(-1);
      }

      // find kernel function
      dpct::kernel_function function;
      checkErrors(DPCT_CHECK_ERROR(DPCT_CHECK_ERROR(
          function = dpct::get_kernel_function(module, "foo"))));

      if (function == NULL) {
        std::cout << "No function\n";
        exit(-1);
      }

      // test cuFuncGetAttribute
      {
        int size;
        int result = DPCT_CHECK_ERROR(
            size =
                dpct::get_kernel_function_info(function).max_work_group_size);
        /*
        DPCT1000:5: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (result != 0) {
          /*
          DPCT1001:4: The statement could not be removed.
          */
          std::cout << "cuFuncGetAttribute failed\n";
          return 1;
        }
      }

      // invoke kernel function
      if (invoke_type == 0) {
        void *args[3] = {&a, &b, &c};
        checkErrors(DPCT_CHECK_ERROR(dpct::invoke_kernel_function(
            function, q_ct1, sycl::range<3>(1, 1, VEC_LENGTH),
            sycl::range<3>(1, 1, 1), 0, args, 0)));
      } else if (invoke_type == 1) {
        int *argBuffer[3] = {a, b, c};
        size_t argBufferSize = sizeof(argBuffer);
        void *extra[] = {
#define EXTRA_BUFFER_ANNOTATIONS
#ifdef EXTRA_BUFFER_ANNOTATIONS
            ((void *)2), &argBufferSize, ((void *)1), argBuffer, ((void *)0)
#else
            ((void *)2), &argBufferSize, ((void *)1), argBuffer, ((void *)0)
#endif
        };

        checkErrors(DPCT_CHECK_ERROR(dpct::invoke_kernel_function(
            function, q_ct1, sycl::range<3>(1, 1, VEC_LENGTH),
            sycl::range<3>(1, 1, 1), 0, 0, extra)));
      }

      dev_ct1.queues_wait_and_throw();

      // check result
      for (int i = 0; i < VEC_LENGTH; i++) {
        if (a[i] != (b[i] * c[i] + seed)) {
          printf("\tRESULT: Failed a[%d]=%d, b[%d]=%d, c[%d]=%d %d\n", i, a[i],
                 i, b[i], i, c[i], seed);
          return 1;
        }
      }
      ++cnt;
      if (cnt & 0x3) {
        // Do not unload every time, so that we exercise automatic unloading on
        // program exit with Windows
        dpct::unload_kernel_library(module);
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  for (auto i = 0; i < 5; i++) {
    if (run_tests())
      return 1;
  }
  return 0;
}
