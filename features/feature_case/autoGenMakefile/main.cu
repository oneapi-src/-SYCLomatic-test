//===--- main.cu -------------------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===------------------------------------------------------ -===//


#include <foo.h>
#include <bar.h>

#include "define.c"
#include "kernel.cu"


int main() {
#ifdef __FOO__
    kernel<<<1,1>>>();
#endif
	return 0;
}