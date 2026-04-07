/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U Linux driver stack, this relies on the
 * ethos-u-linux-driver-stack for hardware interaction.
 */

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

#if defined(EXECUTORCH_ETHOSU_IMX93_PREBUILT)
using EthosU::ETHOSU_UAPI_INFERENCE_MODEL;
using EthosU::ETHOSU_UAPI_NETWORK_BUFFER;
using EthosU::ETHOSU_UAPI_STATUS_OK;
using EthosU::ethosu_uapi_cancel_inference_status;
using EthosU::ethosu_uapi_inference_create;
using EthosU::ethosu_uapi_network_info;
using EthosU::ethosu_uapi_network_create;
using EthosU::ethosu_uapi_result_status;
#endif

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

constexpr int64_t kDefaultEthosUTimeoutNs = 60000000000LL;

struct LinuxDriverOptions {
  std::string device_path = "/dev/ethosu0";
  int64_t timeout_ns = kDefaultEthosUTimeoutNs;
  bool enable_cycle_counter = true;
  std::array<uint32_t, ETHOSU_PMU_EVENT_MAX> pmu_events{};
};

struct PlatformState {
  LinuxDriverOptions options;
};

namespace {

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

template <typename T>
bool read_scalar_value(const CompileSpec& spec, T* out) {
  if (spec.value.buffer == nullptr || spec.value.nbytes != sizeof(T)) {
    return false;
  }
  std::memcpy(out, spec.value.buffer, sizeof(T));
  return true;
}

std::string read_string_value(const CompileSpec& spec) {
  if (spec.value.buffer == nullptr || spec.value.nbytes == 0) {
    return "";
  }
  const char* raw_begin = static_cast<const char*>(spec.value.buffer);
  const char* raw_end = raw_begin + spec.value.nbytes;
  std::string result(raw_begin, raw_end);
  while (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

LinuxDriverOptions parse_linux_options(ArrayRef<CompileSpec> specs) {
  LinuxDriverOptions options;
  constexpr char kDeviceKey[] = "ethosu.device";
  constexpr char kTimeoutKey[] = "ethosu.timeout_ns";
  constexpr char kCycleCounterKey[] = "ethosu.enable_cycle_counter";
  constexpr char kPmuPrefix[] = "ethosu.pmu_event";

  for (const CompileSpec& spec : specs) {
    if (spec.key == nullptr) {
      continue;
    }

    if (strcmp(spec.key, kDeviceKey) == 0) {
      std::string device_path = read_string_value(spec);
      if (!device_path.empty()) {
        options.device_path = device_path;
      }
      continue;
    }

    if (strcmp(spec.key, kTimeoutKey) == 0) {
      int64_t timeout = 0;
      if (read_scalar_value(spec, &timeout) && timeout > 0) {
        options.timeout_ns = timeout;
      }
      continue;
    }

    if (strcmp(spec.key, kCycleCounterKey) == 0) {
      uint8_t enabled = 0;
      if (read_scalar_value(spec, &enabled)) {
        options.enable_cycle_counter = enabled != 0;
      }
      continue;
    }

    if (strncmp(spec.key, kPmuPrefix, strlen(kPmuPrefix)) == 0) {
      const char* index_str = spec.key + strlen(kPmuPrefix);
      char* endptr = nullptr;
      long idx = std::strtol(index_str, &endptr, 10);
      if (endptr != index_str && idx >= 0 &&
          idx < static_cast<long>(ETHOSU_PMU_EVENT_MAX)) {
        uint32_t event = 0;
        if (read_scalar_value(spec, &event)) {
          options.pmu_events[static_cast<size_t>(idx)] = event;
        }
      }
    }
  }

  return options;
}

class EthosULinuxDeviceCache {
 public:
  EthosU::Device& get(const std::string& device_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!device_ || device_path != active_path_) {
      device_ = std::make_unique<EthosU::Device>(device_path.c_str());
      active_path_ = device_path;
    }
    return *device_;
  }

 private:
  std::mutex mutex_;
  std::string active_path_;
  std::unique_ptr<EthosU::Device> device_;
};

EthosULinuxDeviceCache& get_linux_device_cache() {
  static EthosULinuxDeviceCache cache;
  return cache;
}

#if defined(EXECUTORCH_ETHOSU_IMX93_PREBUILT)
void log_memory_layout(const EthosU::MemoryLayout& layout) {
  ET_LOG(
      Info,
      "Ethos-U layout flash_offset=%u arena_offset=%u input_count=%u output_count=%u",
      static_cast<unsigned>(layout.flash_offset),
      static_cast<unsigned>(layout.arena_offset),
      static_cast<unsigned>(layout.input_count),
      static_cast<unsigned>(layout.output_count));

  for (uint32_t i = 0; i < layout.input_count; ++i) {
    ET_LOG(
        Info,
        "Ethos-U layout input[%u] offset=%u size=%u",
        static_cast<unsigned>(i),
        static_cast<unsigned>(layout.input_offset[i]),
        static_cast<unsigned>(layout.input_size[i]));
  }

  for (uint32_t i = 0; i < layout.output_count; ++i) {
    ET_LOG(
        Info,
        "Ethos-U layout output[%u] offset=%u size=%u",
        static_cast<unsigned>(i),
        static_cast<unsigned>(layout.output_offset[i]),
        static_cast<unsigned>(layout.output_size[i]));
  }
}

void log_network_info(const ethosu_uapi_network_info& info) {
  ET_LOG(
      Info,
      "Ethos-U network_info desc='%s' is_vela=%u ifm_count=%u ofm_count=%u",
      info.desc,
      static_cast<unsigned>(info.is_vela),
      static_cast<unsigned>(info.ifm_count),
      static_cast<unsigned>(info.ofm_count));

  for (uint32_t i = 0; i < info.ifm_count; ++i) {
    ET_LOG(
        Info,
        "Ethos-U network_info ifm[%u] size=%u type=%u offset=%u dims=%u",
        static_cast<unsigned>(i),
        static_cast<unsigned>(info.ifm_size[i]),
        static_cast<unsigned>(info.ifm_types[i]),
        static_cast<unsigned>(info.ifm_offset[i]),
        static_cast<unsigned>(info.ifm_dims[i]));
  }

  for (uint32_t i = 0; i < info.ofm_count; ++i) {
    ET_LOG(
        Info,
        "Ethos-U network_info ofm[%u] size=%u type=%u offset=%u dims=%u",
        static_cast<unsigned>(i),
        static_cast<unsigned>(info.ofm_size[i]),
        static_cast<unsigned>(info.ofm_types[i]),
        static_cast<unsigned>(info.ofm_offset[i]),
        static_cast<unsigned>(info.ofm_dims[i]));
  }
}

void log_result_status(const char* label, const ethosu_uapi_result_status& status) {
  ET_LOG(
      Info,
      "%s status=%u cycle_count=%llu pmu=[%u,%u,%u,%u]",
      label,
      static_cast<unsigned>(status.status),
      static_cast<unsigned long long>(status.pmu_count.cycle_count),
      static_cast<unsigned>(status.pmu_count.events[0]),
      static_cast<unsigned>(status.pmu_count.events[1]),
      static_cast<unsigned>(status.pmu_count.events[2]),
      static_cast<unsigned>(status.pmu_count.events[3]));
}
#endif

/*
 * Used for logging when building in Debug mode, but unused building
 * for Release.
 */
[[maybe_unused]] const char* inference_status_to_string(
    EthosU::InferenceStatus status) {
  switch (status) {
    case EthosU::InferenceStatus::OK:
      return "OK";
    case EthosU::InferenceStatus::ERROR:
      return "ERROR";
    case EthosU::InferenceStatus::RUNNING:
      return "RUNNING";
    case EthosU::InferenceStatus::REJECTED:
      return "REJECTED";
    case EthosU::InferenceStatus::ABORTED:
      return "ABORTED";
    case EthosU::InferenceStatus::ABORTING:
      return "ABORTING";
#if !defined(EXECUTORCH_ETHOSU_IMX93_PREBUILT)
    case EthosU::InferenceStatus::PENDING:
      return "PENDING";
#endif
  }
  return "UNKNOWN";
}

Error invoke_linux_driver(
    const VelaHandles& handles,
    const std::vector<const char*>& input_ptrs,
    const std::vector<char*>& output_ptrs,
    const std::vector<size_t>& input_copy_sizes,
    const std::vector<size_t>& output_copy_sizes,
    const LinuxDriverOptions& options) {
  if (handles.outputs == nullptr) {
    ET_LOG(Error, "Ethos-U backend missing output metadata");
    return Error::InvalidProgram;
  }

  try {
    EthosU::Device& device = get_linux_device_cache().get(options.device_path);
#if defined(EXECUTORCH_ETHOSU_IMX93_PREBUILT)
    ET_LOG(
        Info,
        "Ethos-U prebuilt invoke cmd=%zu weight=%zu scratch=%zu inputs=%d outputs=%d",
        handles.cmd_data_size,
        handles.weight_data_size,
        handles.scratch_data_size,
        handles.inputs != nullptr ? handles.inputs->count : 0,
        handles.outputs != nullptr ? handles.outputs->count : 0);
    auto network_buffer =
        std::make_shared<EthosU::Buffer>(device, handles.cmd_data_size);
    network_buffer->resize(handles.cmd_data_size, 0);
    std::memcpy(network_buffer->data(), handles.cmd_data, handles.cmd_data_size);
    ethosu_uapi_network_create network_create{};
    network_create.type = ETHOSU_UAPI_NETWORK_BUFFER;
    network_create.fd = network_buffer->getFd();
    int network_fd =
        device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, static_cast<void*>(&network_create));

    ethosu_uapi_network_info network_info{};
    if (::ioctl(network_fd, ETHOSU_IOCTL_NETWORK_INFO, &network_info) < 0) {
      ET_LOG(
          Error,
          "Failed to query Ethos-U network info: errno=%d (%s)",
          errno,
          std::strerror(errno));
      ::close(network_fd);
      return Error::InvalidState;
    }
    log_network_info(network_info);

    size_t max_io_end = 0;
    for (int i = 0; handles.inputs != nullptr && i < handles.inputs->count; ++i) {
      max_io_end = std::max(
          max_io_end,
          static_cast<size_t>(handles.inputs->io[i].offset) + input_copy_sizes[i]);
    }
    for (int i = 0; handles.outputs != nullptr && i < handles.outputs->count; ++i) {
      max_io_end = std::max(
          max_io_end,
          static_cast<size_t>(handles.outputs->io[i].offset) +
              output_copy_sizes[i]);
    }

    EthosU::MemoryLayout layout{};
    layout.flash_offset = 0;
    layout.arena_offset = align_up(handles.weight_data_size, 16);
    layout.input_count =
        static_cast<uint32_t>(handles.inputs != nullptr ? handles.inputs->count : 0);
    layout.output_count = static_cast<uint32_t>(
        handles.outputs != nullptr ? handles.outputs->count : 0);

    const size_t arena_bytes = std::max(handles.scratch_data_size, max_io_end);
    auto arena_buffer = std::make_shared<EthosU::Buffer>(
        device, layout.arena_offset + arena_bytes);
    arena_buffer->resize(layout.arena_offset + arena_bytes, 0);
    std::memset(arena_buffer->data(), 0, layout.arena_offset + arena_bytes);

    if (handles.weight_data_size > 0) {
      std::memcpy(
          arena_buffer->data() + layout.flash_offset,
          handles.weight_data,
          handles.weight_data_size);
    }

    for (int i = 0; handles.inputs != nullptr && i < handles.inputs->count; ++i) {
      const size_t copy_size = input_copy_sizes[i];
      layout.input_offset[i] = static_cast<uint32_t>(
          layout.arena_offset + static_cast<size_t>(handles.inputs->io[i].offset));
      layout.input_size[i] = static_cast<uint32_t>(copy_size);
      if (copy_size == 0) {
        continue;
      }
      const char* src = input_ptrs[i];
      if (src == nullptr) {
        ET_LOG(Error, "Missing input buffer for index %d", static_cast<int>(i));
        return Error::InvalidState;
      }
      std::memcpy(arena_buffer->data() + layout.input_offset[i], src, copy_size);
    }

    for (int i = 0; handles.outputs != nullptr && i < handles.outputs->count; ++i) {
      layout.output_offset[i] = static_cast<uint32_t>(
          layout.arena_offset + static_cast<size_t>(handles.outputs->io[i].offset));
      layout.output_size[i] = static_cast<uint32_t>(output_copy_sizes[i]);
    }
    log_memory_layout(layout);

    ethosu_uapi_inference_create inference_create{};
    inference_create.ifm_count =
        static_cast<uint32_t>(handles.inputs != nullptr ? handles.inputs->count : 0);
    for (uint32_t i = 0; i < inference_create.ifm_count; ++i) {
      inference_create.ifm_fd[i] =
          static_cast<uint32_t>(arena_buffer->getFd());
    }
    inference_create.ofm_count =
        static_cast<uint32_t>(handles.outputs != nullptr ? handles.outputs->count : 0);
    for (uint32_t i = 0; i < inference_create.ofm_count; ++i) {
      inference_create.ofm_fd[i] =
          static_cast<uint32_t>(arena_buffer->getFd());
    }
    inference_create.memory_layout = layout;
    inference_create.inference_type = ETHOSU_UAPI_INFERENCE_MODEL;
    for (size_t i = 0; i < options.pmu_events.size() &&
         i < ETHOSU_PMU_EVENT_MAX;
         ++i) {
      inference_create.pmu_config.events[i] = options.pmu_events[i];
    }
    inference_create.pmu_config.cycle_count =
        options.enable_cycle_counter ? 1U : 0U;

    int inference_fd =
        ::ioctl(network_fd, ETHOSU_IOCTL_INFERENCE_CREATE, &inference_create);
    if (inference_fd < 0) {
      ET_LOG(
          Error,
          "Failed to create Ethos-U inference: errno=%d (%s)",
          errno,
          std::strerror(errno));
      ::close(network_fd);
      return Error::InvalidState;
    }

    ethosu_uapi_result_status invoke_status{};
    if (::ioctl(inference_fd, ETHOSU_IOCTL_INFERENCE_INVOKE, &invoke_status) < 0) {
      ET_LOG(
          Error,
          "Failed to invoke Ethos-U inference: errno=%d (%s)",
          errno,
          std::strerror(errno));
      ::close(inference_fd);
      ::close(network_fd);
      return Error::InvalidState;
    }
    log_result_status("Ethos-U invoke_status", invoke_status);

    pollfd pfd{};
    pfd.fd = inference_fd;
    pfd.events = POLLIN | POLLERR;
    const int timeout_ms = options.timeout_ns <= 0
        ? -1
        : static_cast<int>(options.timeout_ns / 1000000LL);
    const int poll_ret = ::poll(&pfd, 1, timeout_ms);
    if (poll_ret == 0) {
      ET_LOG(
          Error,
          "Ethos-U inference poll timed out after %lld ns revents=0x%x",
          static_cast<long long>(options.timeout_ns),
          static_cast<unsigned>(pfd.revents));
      ethosu_uapi_result_status timeout_status{};
      if (::ioctl(inference_fd, ETHOSU_IOCTL_INFERENCE_STATUS, &timeout_status) == 0) {
        log_result_status("Ethos-U timeout status before cancel", timeout_status);
      } else {
        ET_LOG(
            Error,
            "Failed to read timeout status: errno=%d (%s)",
            errno,
            std::strerror(errno));
      }
      ethosu_uapi_cancel_inference_status cancel_status{};
      if (::ioctl(inference_fd, ETHOSU_IOCTL_INFERENCE_CANCEL, &cancel_status) == 0) {
        ET_LOG(
            Info,
            "Ethos-U cancel status=%u",
            static_cast<unsigned>(cancel_status.status));
      } else {
        ET_LOG(
            Error,
            "Failed to cancel timed out inference: errno=%d (%s)",
            errno,
            std::strerror(errno));
      }
      ethosu_uapi_result_status after_cancel_status{};
      if (::ioctl(inference_fd, ETHOSU_IOCTL_INFERENCE_STATUS, &after_cancel_status) == 0) {
        log_result_status("Ethos-U timeout status after cancel", after_cancel_status);
      } else {
        ET_LOG(
            Error,
            "Failed to read post-cancel status: errno=%d (%s)",
            errno,
            std::strerror(errno));
      }
      ET_LOG(
          Error,
          "Ethos-U inference timed out after %lld ns",
          static_cast<long long>(options.timeout_ns));
      ::close(inference_fd);
      ::close(network_fd);
      return Error::InvalidState;
    }
    if (poll_ret < 0) {
      ET_LOG(
          Error,
          "Ethos-U inference poll failed errno=%d (%s)",
          errno,
          std::strerror(errno));
      ::close(inference_fd);
      ::close(network_fd);
      return Error::InvalidState;
    }
    ET_LOG(
        Info,
        "Ethos-U inference poll completed revents=0x%x",
        static_cast<unsigned>(pfd.revents));

    ethosu_uapi_result_status status{};
    if (::ioctl(inference_fd, ETHOSU_IOCTL_INFERENCE_STATUS, &status) < 0) {
      ET_LOG(
          Error,
          "Failed to read Ethos-U inference status: errno=%d (%s)",
          errno,
          std::strerror(errno));
      ::close(inference_fd);
      ::close(network_fd);
      return Error::InvalidState;
    }
    log_result_status("Ethos-U final status", status);

    if (status.status != ETHOSU_UAPI_STATUS_OK) {
      ET_LOG(
          Error,
          "Ethos-U inference failed with status %u",
          static_cast<unsigned>(status.status));
      ::close(inference_fd);
      ::close(network_fd);
      return Error::InvalidState;
    }

    if (options.enable_cycle_counter) {
      ET_LOG(
          Info,
          "Ethos-U Linux delegate cycle counter: %llu",
          static_cast<unsigned long long>(status.pmu_count.cycle_count));
    }

    for (int i = 0; i < handles.outputs->count; ++i) {
      const size_t copy_size = output_copy_sizes[i];
      if (copy_size == 0) {
        continue;
      }
      char* dst = output_ptrs[i];
      if (dst == nullptr) {
        ET_LOG(Error, "Missing output buffer for index %d", i);
        return Error::InvalidState;
      }
      std::memcpy(dst, arena_buffer->data() + layout.output_offset[i], copy_size);
    }
    ::close(inference_fd);
    ::close(network_fd);
#else
    auto network = std::make_shared<EthosU::Network>(
        device,
        reinterpret_cast<const unsigned char*>(handles.cmd_data),
        handles.cmd_data_size);

    std::shared_ptr<EthosU::Buffer> constant_buffer =
        std::make_shared<EthosU::Buffer>();
    if (handles.weight_data_size > 0) {
      auto constant_buffers = device.createBuffers({handles.weight_data_size});
      constant_buffer = constant_buffers.front();
      constant_buffer->write(
          const_cast<char*>(handles.weight_data), handles.weight_data_size);
    }

    std::shared_ptr<EthosU::Buffer> intermediate_buffer =
        std::make_shared<EthosU::Buffer>();
    if (handles.scratch_data_size > 0) {
      auto scratch_buffers = device.createBuffers({handles.scratch_data_size});
      intermediate_buffer = scratch_buffers.front();
    }

    std::vector<std::shared_ptr<EthosU::Buffer>> ifm_buffers;
    if (handles.inputs != nullptr && handles.inputs->count > 0) {
      if (input_copy_sizes.size() !=
          static_cast<size_t>(handles.inputs->count)) {
        ET_LOG(
            Error,
            "Mismatch between input metadata (%d) and copy plan (%zu)",
            handles.inputs->count,
            input_copy_sizes.size());
        return Error::InvalidProgram;
      }
      if (input_ptrs.size() != input_copy_sizes.size()) {
        ET_LOG(
            Error,
            "Mismatch between input metadata and runtime pointers (%zu vs %zu)",
            input_ptrs.size(),
            input_copy_sizes.size());
        return Error::InvalidState;
      }
      ifm_buffers = device.createBuffers(input_copy_sizes);
      for (int i = 0; i < handles.inputs->count; ++i) {
        const size_t copy_size = input_copy_sizes[i];
        if (copy_size == 0) {
          continue;
        }
        const char* src = input_ptrs[i];
        if (src == nullptr) {
          ET_LOG(Error, "Missing input buffer for index %d", i);
          return Error::InvalidState;
        }
        ifm_buffers[i]->write(const_cast<char*>(src), copy_size);
      }
    }

    if (output_copy_sizes.size() !=
        static_cast<size_t>(handles.outputs->count)) {
      ET_LOG(
          Error,
          "Mismatch between output metadata (%d) and copy plan (%zu)",
          handles.outputs->count,
          output_copy_sizes.size());
      return Error::InvalidProgram;
    }
    if (output_ptrs.size() != output_copy_sizes.size()) {
      ET_LOG(
          Error,
          "Mismatch between output metadata and runtime buffers (%zu vs %zu)",
          output_ptrs.size(),
          output_copy_sizes.size());
      return Error::InvalidState;
    }
    auto ofm_buffers = device.createBuffers(output_copy_sizes);

    auto inference = std::make_unique<EthosU::Inference>(
        network,
        ifm_buffers.begin(),
        ifm_buffers.end(),
        ofm_buffers.begin(),
        ofm_buffers.end(),
        intermediate_buffer,
        constant_buffer,
        options.pmu_events,
        options.enable_cycle_counter);

    if (inference->wait(options.timeout_ns)) {
      ET_LOG(
          Error,
          "Ethos-U inference timed out after %lld ns",
          static_cast<long long>(options.timeout_ns));
      return Error::InvalidState;
    }

    auto status = inference->status();
    if (status != EthosU::InferenceStatus::OK) {
      ET_LOG(
          Error,
          "Ethos-U inference failed with status %s",
          inference_status_to_string(status));
      return Error::InvalidState;
    }

    if (options.enable_cycle_counter) {
      try {
        ET_LOG(
            Info,
            "Ethos-U Linux delegate cycle counter: %llu",
            static_cast<unsigned long long>(inference->getCycleCounter()));
      } catch (const std::exception& e) {
        ET_LOG(Debug, "Failed to read Ethos-U cycle counter: %s", e.what());
      }
    }

    for (int i = 0; i < handles.outputs->count; ++i) {
      const size_t copy_size = output_copy_sizes[i];
      if (copy_size == 0) {
        continue;
      }
      char* dst = output_ptrs[i];
      if (dst == nullptr) {
        ET_LOG(Error, "Missing output buffer for index %d", i);
        return Error::InvalidState;
      }
      ofm_buffers[i]->read(dst, copy_size);
    }
#endif
  } catch (const std::exception& e) {
    ET_LOG(Error, "Ethos-U Linux driver invocation failed: %s", e.what());
    return Error::InvalidState;
  }

  return Error::Ok;
}
} // namespace

PlatformState* platform_init(
    ArrayRef<CompileSpec> specs,
    MemoryAllocator* allocator) {
  (void)allocator;
  PlatformState* state = new (std::nothrow) PlatformState();
  if (state == nullptr) {
    return nullptr;
  }
  state->options = parse_linux_options(specs);
  return state;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* execution_handle,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* /*ethosu_scratch*/) {
  std::vector<size_t> input_copy_sizes(input_count, 0);
  std::vector<const char*> linux_input_ptrs(input_count, nullptr);

  std::vector<size_t> output_io_bytes(output_count, 0);
  std::vector<char*> linux_output_ptrs(output_count, nullptr);
  std::vector<std::vector<char>> output_scratch_buffers(output_count);
  std::vector<bool> output_needs_adjustment(output_count, false);

  for (int i = 0; i < input_count; ++i) {
    auto tensor_in = args[i]->toTensor();
    linux_input_ptrs[i] = tensor_in.const_data_ptr<char>();
    input_copy_sizes[i] = tensor_in.nbytes();
  }

  if (handles.outputs != nullptr) {
    for (int i = 0; i < output_count; ++i) {
      int tensor_count = 1, io_count = 1;
      auto tensor_out = args[input_count + i]->toTensor();
      calculate_dimensions(
          tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);
      if (i < static_cast<int>(output_io_bytes.size())) {
        output_io_bytes[i] = static_cast<size_t>(io_count) *
            static_cast<size_t>(handles.outputs->io[i].elem_size);
      }
      const size_t tensor_nbytes = tensor_out.nbytes();
      if (i < static_cast<int>(output_io_bytes.size()) &&
          output_io_bytes[i] != tensor_nbytes) {
        output_scratch_buffers[i].resize(output_io_bytes[i]);
        linux_output_ptrs[i] = output_scratch_buffers[i].data();
        output_needs_adjustment[i] = true;
      } else {
        linux_output_ptrs[i] = tensor_out.mutable_data_ptr<char>();
      }
    }
  }

  const PlatformState* state = execution_handle->platform_state;
  if (state == nullptr) {
    ET_LOG(Error, "Ethos-U Linux backend missing platform state");
    return Error::InvalidState;
  }

  Error status = invoke_linux_driver(
      handles,
      linux_input_ptrs,
      linux_output_ptrs,
      input_copy_sizes,
      output_io_bytes,
      state->options);
  if (status != Error::Ok) {
    return status;
  }

  if (handles.outputs != nullptr) {
    for (int i = 0; i < output_count; ++i) {
      if (!output_needs_adjustment[i]) {
        continue;
      }
      auto tensor_out = args[input_count + i]->toTensor();
      const size_t tensor_nbytes = tensor_out.nbytes();
      Error adjust_status = copy_with_layout_adjustment(
          handles.outputs->io[i],
          i,
          output_scratch_buffers[i].data(),
          tensor_out,
          tensor_nbytes);
      if (adjust_status != Error::Ok) {
        return adjust_status;
      }
    }
  }

  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
