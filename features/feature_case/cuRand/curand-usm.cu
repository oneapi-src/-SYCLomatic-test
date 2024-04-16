// ====------ curand-usm.cu---------- *- CUDA -* ----===////
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

  curandSetGeneratorOrdering(rng, CURAND_ORDERING_PSEUDO_BEST);
  s1 = curandSetGeneratorOrdering(rng2, CURAND_ORDERING_PSEUDO_DEFAULT);

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

class B{
public:
  B(){
    curandCreateGenerator(&rng, CURAND_RNG_QUASI_DEFAULT);
    curandSetQuasiRandomGeneratorDimensions(rng, 1243);
    cudaMalloc(&karg1, 32 * sizeof(int));
  }
  ~B(){
    curandDestroyGenerator(rng);
    cudaFree(karg1);
  }
private:
  curandGenerator_t rng;
  int *karg1;
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

int bar5(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar6(float *x_gpu, size_t n) {
  static curandGenerator_t gen[16];
  static int init[16] = {0};
  int i = 0;
  if(!init[i]) {
    curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen[i], 1234);
    init[i] = 1;
  }
  curandGenerateUniform(gen[i], x_gpu, n);
}
