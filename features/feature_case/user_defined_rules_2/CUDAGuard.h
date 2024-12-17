#ifndef __CUDAGUARD_H__
#define __CUDAGUARD_H__

#include "c10/core/Device.h"
#include <optional>
#include <string>
namespace c10 {
namespace cuda {
class OptionalCUDAGuard {
public:
  OptionalCUDAGuard(std::optional<c10::Device> device) {}
};
} // namespace cuda
} // namespace c10

#endif // __CUDAGUARD_H__
