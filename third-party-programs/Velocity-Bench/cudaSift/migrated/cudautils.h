#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#define DPCT_COMPAT_RT_VERSION 12020
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <iostream>
#include <chrono>

#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync() __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(dpct::err0 err, const char *file, const int line)
{
}

inline void __safeThreadSync(const char *file, const int line) try {
  dpct::err0 err =
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  /*
  DPCT1010:86: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 err = 0;
}

inline bool deviceInit(int dev)
{
  int deviceCount;
  safeCall(
      DPCT_CHECK_ERROR(deviceCount = dpct::dev_mgr::instance().device_count()));
  if (deviceCount == 0)
  {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    return false;
  }
  if (dev < 0)
    dev = 0;
  if (dev > deviceCount - 1)
    dev = deviceCount - 1;
  dpct::device_info deviceProp;
  safeCall(DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(dev))));
  /*
  DPCT1005:88: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if (deviceProp.get_major_version() < 1)
  {
    fprintf(stderr, "error: device does not support CUDA.\n");
    return false;
  }
  /*
  DPCT1093:89: The "dev" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  safeCall(DPCT_CHECK_ERROR(dpct::select_device(dev)));
  return true;
}

class TimerGPU
{
public:
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  dpct::queue_ptr stream;
  TimerGPU(dpct::queue_ptr stream_ = &dpct::get_in_order_queue())
      : stream(stream_)
  {
    start = new sycl::event();
    stop = new sycl::event();
    /*
    DPCT1012:90: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
    *start = stream->ext_oneapi_submit_barrier();
  }
  ~TimerGPU()
  {
    dpct::destroy_event(start);
    dpct::destroy_event(stop);
  }
  float read()
  {
    /*
    DPCT1012:91: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    *stop = stream->ext_oneapi_submit_barrier();
    stop->wait_and_throw();
    float time;
    time =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    return time;
  }
};

class TimerCPU
{
  static const int bits = 10;

public:
  long long beg_clock;
  float freq;
  TimerCPU(float freq_) : freq(freq_)
  { // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits)
  {
#ifdef WIN32
    return __rdtsc() / (1LL << bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31"
            : "=a"(low), "=d"(high));
    return ((long long)high << (32 - bits)) | ((long long)low >> bits);
#endif
  }
  float read()
  {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
    return time;
  }
};

template <class T>
__inline__ T ShiftDown(T var, unsigned int delta,
                       const sycl::nd_item<3> &item_ct1, int width = 32)
{
#if (DPCT_COMPAT_RT_VERSION >= 9000)
  /*
  DPCT1023:0: The SYCL sub-group does not support mask options for
  dpct::shift_sub_group_left. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_down_sync.
  */
  /*
  DPCT1096:225: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::shift_sub_group_left" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  return dpct::shift_sub_group_left(item_ct1.get_sub_group(), var, delta,
                                    width);
#else
  return __shfl_down(var, delta, width);
#endif
}

template <class T>
__inline__ T ShiftUp(T var, unsigned int delta,
                     const sycl::nd_item<3> &item_ct1, int width = 32)
{
#if (DPCT_COMPAT_RT_VERSION >= 9000)
  /*
  DPCT1023:1: The SYCL sub-group does not support mask options for
  dpct::shift_sub_group_right. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_up_sync.
  */
  return dpct::shift_sub_group_right(item_ct1.get_sub_group(), var, delta,
                                     width);
#else
  return __shfl_up(var, delta, width);
#endif
}

template <class T>
__inline__ T Shuffle(T var, unsigned int lane, const sycl::nd_item<3> &item_ct1, int width = 32)
{
#if (DPCT_COMPAT_RT_VERSION >= 9000)
  /*
  DPCT1023:2: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_sync.
  */
  return dpct::select_from_sub_group(item_ct1.get_sub_group(), var, lane,
                                     width);
#else
  return __shfl(var, lane, width);
#endif
}

#endif
