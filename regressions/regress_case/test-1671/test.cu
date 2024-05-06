// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<iostream>
#include<cuda_runtime.h>
#include<stdio.h>



__global__ void mykernel(unsigned int *dev){

extern __shared__ unsigned int sm[];
unsigned int* as= (unsigned int*)sm;

const int kc=threadIdx.x;
 const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    atomicOr(&as[kc], (unsigned int)1);

dev[tid]=as[kc];

}




int main(){
   unsigned int N=10;
   unsigned int *h_array=(unsigned int*)malloc(N*sizeof(unsigned int));


   for(int i=0;i<N;i++)
        h_array[i]=0;

   unsigned int *d_array1;
   cudaMalloc((void**)&d_array1,N*sizeof(unsigned int));

   cudaMemcpy((void*)d_array1,(void*)h_array,N*sizeof(int),cudaMemcpyHostToDevice);


   unsigned int sm_size;
   sm_size= N * sizeof(unsigned int);

   mykernel<<<1,N,sm_size>>>(d_array1);

   cudaMemcpy(h_array,d_array1,N*sizeof(int),cudaMemcpyDeviceToHost);

   printf("The values in h_retri_array1 is:\n");
   for(int i=0;i<N;i++)
       printf("%d\n",h_array[i]);

   return 0;

}
