// ====------ devicemgr_multi_thread_aware.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

// This test case creates NUM_THREADS threads, and each thread will use a
// unique sycl device to do some job.
#define NUM_THREADS 2

void thread_using_device(unsigned long long &q/*in: device id,  out: queue ptr*/) {
  unsigned int tid = dpct::get_tid();
  unsigned int did = q;//input: q is device id.
  printf("Thread id %u, using device: %d\n", tid, did);
  unsigned int dev_cnt = dpct::dev_mgr::instance().device_count();
  printf(" sycl device cnt:%d\n", dev_cnt);
  if (did >= dev_cnt) {
    printf("error: deivce id is out of range: deviceid:%d >= dev_cnt:%d\n", did,
           dev_cnt);
    q = 0;
    exit(-1);
  }
  // test include: setlect the device, get the device info,  run a kernel
  if (did != 0) {
    dpct::dev_mgr::instance().select_device(did);
  }
  if (dpct::dev_mgr::instance().current_device_id() != did) {
    printf(" testfail\n");
    q = 0;
    exit(-1);
  }
  printf(" run kernel:...\n");
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(10, 1, 1), sycl::range<3>(2, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          int i;
          i++;
        });
  }

  );
  dpct::get_default_queue().wait();
  printf("ret:%p\n", &dpct::get_default_queue());
  q = (unsigned long long)&dpct::get_default_queue();
}

int main(int argc, char *argv[]) {
  std::vector<std::thread> threads(NUM_THREADS);
  int retcode;
  int t;
  unsigned int dev_cnt = dpct::dev_mgr::instance().device_count();
  printf("In main:sycl device cnt:%d\n", dev_cnt);
  if (dev_cnt < NUM_THREADS) {
    printf(
        "only %d sycl deveice available, need %d sycl device for this test\n",
        dev_cnt, NUM_THREADS);
    return -1;
  }

  unsigned long long queue_ptr[NUM_THREADS];
  for (t = 0; t < NUM_THREADS; t++) {
    printf("In main: creating thread %d\n", t);
    queue_ptr[t] = t;
    threads[t] = std::thread(thread_using_device, std::ref(queue_ptr[t]));
  }

  for (auto &thread : threads) {
    thread.join();
  }
  // here check the default_queue retrun from each child is differnt.
  if ((queue_ptr[0] != NULL) && (queue_ptr[1] != NULL) &&
      (queue_ptr[0] != queue_ptr[1])) {
    printf("testpass\n");
    return 0;
  } else {
    printf("failed\n");
    return -1;
  }
}
