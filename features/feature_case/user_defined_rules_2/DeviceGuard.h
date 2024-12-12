#include "c10/core/Device.h"
#include <optional>
#include <string>
namespace c10 {
class OptionalDeviceGuard {
public:
  OptionalDeviceGuard(std::optional<c10::Device> device) {}
};
} // namespace c10
