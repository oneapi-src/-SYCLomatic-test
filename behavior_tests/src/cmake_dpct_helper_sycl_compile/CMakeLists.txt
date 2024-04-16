# ==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation SPDX-License-Identifier: Apache-2.0 WITH
# LLVM-exception See https://llvm.org/LICENSE.txt for license information.
#
# ===----------------------------------------------------------------------===//
cmake_minimum_required(VERSION 3.10)

project(cuda_compile_sample)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

find_program(
  DPCT_BIN_PATH
  NAMES dpct
  PATHS)

get_filename_component(DPCT_BIN_DIR ${DPCT_BIN_PATH} DIRECTORY)
set(DPCT_CMAKE_PATH "${DPCT_BIN_DIR}/../cmake/dpct.cmake")
include(${DPCT_CMAKE_PATH})

dpct_helper_sycl_compile(OBJSETONE a.dp.cpp b.dp.cpp)

set(_SRC_FILES c.dp.cpp d.dp.cpp)
dpct_helper_sycl_compile(OBJSETTWO ${_SRC_FILES})

add_executable(run main.dp.cpp)

if(WIN32)
  set(SHARED_LIB_EXTN ".lib")
elseif(UNIX AND NOT APPLE)
  set(SHARED_LIB_EXTN ".so")
else()
  message(FATAL "Unsupported platform")
endif()

set(LIB_COUNTER 1)
set(LIBPATHS "")

# Iterate of each object file and "build" the shared library out of it. Also
# create an list of library path for the cmake target tdc
foreach(OBJPATH IN LISTS OBJSETONE OBJSETTWO)
  # build the library path by replacing the extension to SHARED_LIB_EXTN
  string(REGEX REPLACE "${CMAKE_CXX_OUTPUT_EXTENSION}$" ${SHARED_LIB_EXTN} LIBPATH ${OBJPATH})
  set (LIBNAME lib${LIB_COUNTER})
  if (UNIX AND NOT APPLE)
    # copy the .o file another file with .so extension
    add_custom_command(
      OUTPUT  ${LIBPATH}
      COMMAND ${CMAKE_COMMAND} -E copy ${OBJPATH} ${LIBPATH}
      DEPENDS ${OBJPATH}
      COMMENT "Generating ${LIBNAME}")
  elseif(WIN32)
    # In case of Windows .lib files are generated along with .obj file if
    # dllexport is present. But cmake complains about missing rule to build .lib
    # file. To make cmake happy add a dummy command.
    add_custom_command(
      OUTPUT  ${LIBPATH}
      COMMAND ${CMAKE_COMMAND} -E echo "Generating ${LIBNAME}"
      DEPENDS ${OBJPATH})
  endif()

  list(APPEND LIBPATHS ${LIBPATH})
  # let cmake know that we have library to generate
  add_library(${LIBNAME} SHARED IMPORTED)

  # for the library generated above set path to actual library in filesystem
  if (WIN32)
    set_target_properties(${LIBNAME} PROPERTIES IMPORTED_IMPLIB ${LIBPATH})
  elseif (UNIX AND NOT APPLE)
    set_target_properties(${LIBNAME} PROPERTIES IMPORTED_LOCATION ${LIBPATH})
  endif()

  # inform cmake to link to the above generated library
  target_link_libraries(run PRIVATE ${LIBNAME})
  MATH(EXPR LIB_COUNTER "${LIB_COUNTER}+1")
endforeach()

# create custom target which relates to the above generated library paths
add_custom_target(libGenerator ALL DEPENDS ${LIBPATHS})

# inform cmake that the executable depends on the above generated custom target
add_dependencies(run libGenerator)
