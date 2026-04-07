#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <uapi/ethosu.h>

#define DEFAULT_ARENA_SIZE_OF_MB 16

#undef major
#undef minor

namespace EthosU {

typedef ethosu_uapi_memory_layout MemoryLayout;

class Exception : public std::exception {
 public:
  explicit Exception(const char* msg);
  ~Exception() throw() override;
  const char* what() const throw() override;
};

class SemanticVersion {
 public:
  SemanticVersion(
      uint32_t major_ = 0,
      uint32_t minor_ = 0,
      uint32_t patch_ = 0)
      : major(major_), minor(minor_), patch(patch_) {}

  bool operator==(const SemanticVersion& other);
  bool operator<(const SemanticVersion& other);
  bool operator<=(const SemanticVersion& other);
  bool operator!=(const SemanticVersion& other);
  bool operator>(const SemanticVersion& other);
  bool operator>=(const SemanticVersion& other);

  uint32_t major;
  uint32_t minor;
  uint32_t patch;
};

std::ostream& operator<<(std::ostream& out, const SemanticVersion& v);

struct HardwareId {
  HardwareId(
      uint32_t version_status_,
      const SemanticVersion& version_,
      const SemanticVersion& product_,
      const SemanticVersion& architecture_)
      : versionStatus(version_status_),
        version(version_),
        product(product_),
        architecture(architecture_) {}

  uint32_t versionStatus;
  SemanticVersion version;
  SemanticVersion product;
  SemanticVersion architecture;
};

struct HardwareConfiguration {
  HardwareConfiguration(
      uint32_t macs_per_clock_cycle_,
      uint32_t cmd_stream_version_,
      bool custom_dma_)
      : macsPerClockCycle(macs_per_clock_cycle_),
        cmdStreamVersion(cmd_stream_version_),
        customDma(custom_dma_) {}

  uint32_t macsPerClockCycle;
  uint32_t cmdStreamVersion;
  bool customDma;
};

class Capabilities {
 public:
  Capabilities(
      const HardwareId& hw_id_,
      const HardwareConfiguration& hw_cfg_,
      const SemanticVersion& driver_)
      : hwId(hw_id_), hwCfg(hw_cfg_), driver(driver_) {}

  HardwareId hwId;
  HardwareConfiguration hwCfg;
  SemanticVersion driver;
};

class Device {
 public:
  explicit Device(const char* device = "/dev/ethosu0");
  virtual ~Device() noexcept(false);

  int ioctl(unsigned long cmd, void* data = nullptr) const;
  Capabilities capabilities() const;

 private:
  int fd;
};

class Buffer {
 public:
  Buffer(const Device& device, size_t capacity);
  virtual ~Buffer() noexcept(false);

  size_t capacity() const;
  void clear() const;
  char* data() const;
  void resize(size_t size, size_t offset = 0) const;
  size_t offset() const;
  size_t size() const;
  int getFd() const;

 private:
  int fd;
  char* dataPtr;
  const size_t dataCapacity;
};

class Network {
 public:
  Network(const Device& device, std::shared_ptr<Buffer>& buffer);
  Network(const Device& device, unsigned index);
  virtual ~Network() noexcept(false);

  int ioctl(unsigned long cmd, void* data = nullptr);
  std::shared_ptr<Buffer> getBuffer();
  const std::vector<size_t>& getIfmDims() const;
  const std::vector<size_t>& getOfmDims() const;
  size_t getInputCount() const;
  size_t getOutputCount() const;
  int32_t getInputDataOffset(int index);
  int32_t getOutputDataOffset(int index);
  const std::vector<std::vector<size_t>>& getIfmShapes() const;
  const std::vector<std::vector<size_t>>& getOfmShapes() const;
  const std::vector<int>& getIfmTypes() const;
  const std::vector<int>& getOfmTypes() const;
  const Device& getDevice() const;
  bool isVelaModel() const;

 private:
  void collectNetworkInfo();

  int fd;
  std::shared_ptr<Buffer> buffer;
  std::vector<size_t> ifmDims;
  std::vector<size_t> ofmDims;
  std::vector<int32_t> ifmDataOffset;
  std::vector<int32_t> ofmDataOffset;
  std::vector<std::vector<size_t>> ifmShapes;
  std::vector<std::vector<size_t>> ofmShapes;
  std::vector<int> ifmTypes;
  std::vector<int> ofmTypes;
  const Device& device;
  bool is_vela_model;
};

enum class InferenceStatus {
  OK,
  ERROR,
  RUNNING,
  REJECTED,
  ABORTED,
  ABORTING,
};

std::ostream& operator<<(std::ostream& out, const InferenceStatus& status);

class Inference {
 public:
  template <typename T, typename U>
  Inference(
      const std::shared_ptr<Network>& network,
      const T& arena_buffer,
      const U& counters,
      bool enable_cycle_counter,
      MemoryLayout layout)
      : network(network), arenaBuffer(arena_buffer) {
    std::vector<uint32_t> counter_configs = initializeCounterConfig();
    if (counters.size() > counter_configs.size()) {
      throw EthosU::Exception("PMU Counters argument too large.");
    }
    std::copy(counters.begin(), counters.end(), counter_configs.begin());
    create(counter_configs, enable_cycle_counter, layout);
  }

  virtual ~Inference() noexcept(false);

  bool wait(int64_t timeout_nanos = -1) const;
  bool invoke(int64_t timeout_nanos = -1) const;
  const std::vector<uint64_t> getPmuCounters() const;
  uint64_t getCycleCounter() const;
  bool cancel() const;
  InferenceStatus status() const;
  int getFd() const;
  const std::shared_ptr<Network> getNetwork() const;
  std::shared_ptr<Buffer>& getArenaBuffer();
  static uint32_t getMaxPmuEventCounters();

  char* getInputData(int index = 0);
  char* getOutputData(int index = 0);

 private:
  void create(
      std::vector<uint32_t>& counter_configs,
      bool enable_cycle_counter,
      MemoryLayout layout);
  static std::vector<uint32_t> initializeCounterConfig();

  int fd;
  const std::shared_ptr<Network> network;
  std::shared_ptr<Buffer> arenaBuffer;
};

struct TensorInfo {
  int type;
  std::vector<size_t> shape;
};

} // namespace EthosU
