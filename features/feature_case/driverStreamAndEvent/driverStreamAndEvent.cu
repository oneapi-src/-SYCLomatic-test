// ====------ driverStreamAndEvent.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include<vector>
#include<cuda.h>

void process(CUstream st, char *data, CUresult status) {}

template<typename T>
void callback(CUstream hStream, CUresult status, void* userData) {
  T *data = static_cast<T *>(userData);
  process(hStream, data, status);
}

int driverStreamQuery(void);

int main(){
  CUfunction f;
  CUstream s;
  cudaStreamCreate(&s);
  CUevent e;
  CUdeviceptr  cuPtr;
  void* data;
  unsigned int flag;
  cuStreamAddCallback(s, callback<char>, data, flag);

  cuFuncSetCacheConfig(f, CU_FUNC_CACHE_PREFER_NONE);

  cuStreamCreate(&s, CU_STREAM_DEFAULT);
  cuStreamSynchronize(s);

  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

  cuEventRecord(e, s);
  cuEventSynchronize(e);

  CUresult r;
  r = cuEventQuery(e);

  CUevent start, end;
  cuEventCreate(&start, CU_EVENT_DEFAULT);
  cuEventCreate(&end, CU_EVENT_DEFAULT);

  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);

  int rr;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);

  cuStreamAttachMemAsync(s, cuPtr, std::vector<int>(1,1).front(), flag);

  cuStreamDestroy(s);
  cuEventDestroy(start);
  cuEventDestroy(end);

  return driverStreamQuery();
}

