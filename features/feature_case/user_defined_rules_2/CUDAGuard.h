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
