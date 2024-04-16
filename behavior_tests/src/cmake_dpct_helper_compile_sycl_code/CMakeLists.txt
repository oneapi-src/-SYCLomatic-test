#==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# See https://llvm.org/LICENSE.txt for license information.
#
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

project(cmake_ptx NONE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
find_program(
  dpct_bin_path
  NAMES dpct
  PATHS)
get_filename_component(bin_path_of_dpct ${dpct_bin_path} DIRECTORY)
set(dpct_cmake_file_path "${bin_path_of_dpct}/../cmake/dpct.cmake")
include(${dpct_cmake_file_path})

dpct_helper_compile_sycl_code(cuda_ptx_files jit.dp.cpp)

set(_generated_files "")

set(options "")
if (UNIX)
  set(options -lstdc++fs)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/app.run
  COMMAND icpx -fsycl -o ${CMAKE_CURRENT_BINARY_DIR}/app.run
          ${CMAKE_SOURCE_DIR}/kernel_library.cpp.dp.cpp ${options}
  DEPENDS ${CMAKE_SOURCE_DIR}/kernel_library.cpp.dp.cpp
  COMMENT "Generating app.run")

list(APPEND _generated_files ${CMAKE_CURRENT_BINARY_DIR}/app.run)

set(_copy_files "")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/premade.ptx
  COMMAND ${CMAKE_COMMAND} -E copy ${cuda_ptx_files}
          ${CMAKE_CURRENT_BINARY_DIR}/premade.ptx)
list(APPEND _copy_files ${CMAKE_CURRENT_BINARY_DIR}/premade.ptx)

add_custom_target(ptx ALL DEPENDS ${_generated_files} ${cuda_ptx_files}
                                  ${_copy_files})
