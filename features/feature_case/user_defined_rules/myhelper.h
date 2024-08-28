size_t *foo4(int ***i, int *j, sycl::queue q, sycl::context c,
#if USE_DPCT_HELPER
             dpct::device_ext &d
#else
             syclcompat::device_ext &d
#endif
) {
  return 0;
}

class mytype {};
