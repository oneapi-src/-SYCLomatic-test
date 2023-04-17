// ====------ curand.cu---------- *- CUDA -* ----===////
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

int main(){
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  curandGenerateUniform(rng, d_data, 100*100);

  s1 = curandGenerateUniform(rng, d_data, 100*100);

  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  curandGenerateUniformDouble(rng, d_data_d, 100*100);

  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);

  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;

  s1 = curandGenerate(rng, d_data_ui, 100*100);

  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  cudaStream_t stream;
  curandSetStream(rng, stream);

  curandDestroyGenerator(rng);
  s1 = curandDestroyGenerator(rng);
}

curandStatus_t foo1();
curandStatus foo2();

class A{
public:
  A(){
    curandCreateGenerator(&rng, CURAND_RNG_QUASI_DEFAULT);
    curandSetQuasiRandomGeneratorDimensions(rng, 1243);
  }
  ~A(){
    curandDestroyGenerator(rng);
  }
private:
  curandGenerator_t rng;
};



void bar1(){
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

void bar3(){
  curandGenerator_t rng;
  curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
  float *d_data;
  curandErrCheck(curandGenerateUniform(rng, d_data, 100*100));
  curandErrCheck(curandDestroyGenerator(rng));
}

void bar4(){
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}


int bar6(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar7() {
  curandGenerator_t rng;
  curandRngType_t rngT1 = CURAND_RNG_PSEUDO_DEFAULT;
  curandRngType_t rngT2 = CURAND_RNG_PSEUDO_XORWOW;
  curandRngType_t rngT3 = CURAND_RNG_PSEUDO_MRG32K3A;
  curandRngType_t rngT4 = CURAND_RNG_PSEUDO_MTGP32;
  curandRngType_t rngT5 = CURAND_RNG_PSEUDO_MT19937;
  curandRngType_t rngT6 = CURAND_RNG_PSEUDO_PHILOX4_32_10;
  curandRngType_t rngT7 = CURAND_RNG_QUASI_DEFAULT;
  curandRngType_t rngT8 = CURAND_RNG_QUASI_SOBOL32;
  curandRngType_t rngT9 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
  curandRngType_t rngT10 = CURAND_RNG_QUASI_SOBOL64;
  curandRngType_t rngT11 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  curandCreateGeneratorHost(&rng, rngT1);
}
