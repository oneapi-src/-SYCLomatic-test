// ====------ curandEnum.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

curandStatus_t foo(
  curandStatus_t a1,
  curandStatus_t a2,
  curandStatus_t a3,
  curandStatus_t a4,
  curandStatus_t a5,
  curandStatus_t a6,
  curandStatus_t a7,
  curandStatus_t a8,
  curandStatus_t a9,
  curandStatus_t a10,
  curandStatus_t a11,
  curandStatus_t a12,
  curandStatus_t a13) {}

curandOrdering_t goo(curandOrdering_t b1, curandOrdering_t b2,
                     // curandOrdering_t b3,
                     curandOrdering_t b4, curandOrdering_t b5
                     // , curandOrdering_t b6
) {
  return b1;
}

int main() {
  curandStatus_t a1 = CURAND_STATUS_SUCCESS;
  curandStatus_t a2 = CURAND_STATUS_VERSION_MISMATCH;
  curandStatus_t a3 = CURAND_STATUS_NOT_INITIALIZED;
  curandStatus_t a4 = CURAND_STATUS_ALLOCATION_FAILED;
  curandStatus_t a5 = CURAND_STATUS_TYPE_ERROR;
  curandStatus_t a6 = CURAND_STATUS_OUT_OF_RANGE;
  curandStatus_t a7 = CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  curandStatus_t a8 = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
  curandStatus_t a9 = CURAND_STATUS_LAUNCH_FAILURE;
  curandStatus_t a10 = CURAND_STATUS_PREEXISTING_FAILURE;
  curandStatus_t a11 = CURAND_STATUS_INITIALIZATION_FAILED;
  curandStatus_t a12 = CURAND_STATUS_ARCH_MISMATCH;
  curandStatus_t a13 = CURAND_STATUS_INTERNAL_ERROR;


  foo(
    CURAND_STATUS_SUCCESS,
    CURAND_STATUS_VERSION_MISMATCH,
    CURAND_STATUS_NOT_INITIALIZED,
    CURAND_STATUS_ALLOCATION_FAILED,
    CURAND_STATUS_TYPE_ERROR,
    CURAND_STATUS_OUT_OF_RANGE,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
    CURAND_STATUS_LAUNCH_FAILURE,
    CURAND_STATUS_PREEXISTING_FAILURE,
    CURAND_STATUS_INITIALIZATION_FAILED,
    CURAND_STATUS_ARCH_MISMATCH,
    CURAND_STATUS_INTERNAL_ERROR);

  curandOrdering_t b1 = CURAND_ORDERING_PSEUDO_BEST;
  curandOrdering_t b2 = CURAND_ORDERING_PSEUDO_DEFAULT;
  // curandOrdering_t b3 = CURAND_ORDERING_PSEUDO_SEEDED;
  curandOrdering_t b4 = CURAND_ORDERING_PSEUDO_LEGACY;
  curandOrdering_t b5 = CURAND_ORDERING_PSEUDO_DYNAMIC;
  // curandOrdering_t b6 = CURAND_ORDERING_QUASI_DEFAULT;

  goo(CURAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_DEFAULT,
      // CURAND_ORDERING_PSEUDO_SEEDED,
      CURAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_DYNAMIC
      // , CURAND_ORDERING_QUASI_DEFAULT
  );
}
