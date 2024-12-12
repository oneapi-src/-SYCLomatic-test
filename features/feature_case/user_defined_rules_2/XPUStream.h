#include <sycl/sycl.hpp>

namespace c10 {
namespace xpu {
class XPUStream_t {
public:
  XPUStream_t() { _q_ptr = &_q; }
  sycl::queue &queue() const { return *_q_ptr; }
  operator sycl::queue &() const { return *_q_ptr; }

private:
  sycl::queue *_q_ptr;
  sycl::queue _q;
};
XPUStream_t getCurrentXPUStream() {
  static XPUStream_t stream;
  return stream;
}
} // namespace xpu
} // namespace c10
