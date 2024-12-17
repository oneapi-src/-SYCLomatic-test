#ifndef __DEVICE_GUARD_H__
#define __DEVICE_GUARD_H__

#include "c10/core/Device.h"
#include <optional>
#include <string>
namespace c10 {
class OptionalDeviceGuard {
public:
  OptionalDeviceGuard(std::optional<c10::Device> device) {}
};
} // namespace c10

#endif // __DEVICE_GUARD_H__
