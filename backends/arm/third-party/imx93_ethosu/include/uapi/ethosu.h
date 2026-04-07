#pragma once

#include <linux/ioctl.h>
#include <linux/types.h>

#ifdef __cplusplus
namespace EthosU {
#endif

#define ETHOSU_IOCTL_BASE 0x01
#define ETHOSU_IO(nr) _IO(ETHOSU_IOCTL_BASE, nr)
#define ETHOSU_IOR(nr, type) _IOR(ETHOSU_IOCTL_BASE, nr, type)
#define ETHOSU_IOW(nr, type) _IOW(ETHOSU_IOCTL_BASE, nr, type)
#define ETHOSU_IOWR(nr, type) _IOWR(ETHOSU_IOCTL_BASE, nr, type)

#define ETHOSU_IOCTL_PING ETHOSU_IO(0x00)
#define ETHOSU_IOCTL_VERSION_REQ ETHOSU_IO(0x01)
#define ETHOSU_IOCTL_CAPABILITIES_REQ \
  ETHOSU_IOR(0x02, struct ethosu_uapi_device_capabilities)
#define ETHOSU_IOCTL_BUFFER_CREATE ETHOSU_IOR(0x10, struct ethosu_uapi_buffer_create)
#define ETHOSU_IOCTL_BUFFER_SET ETHOSU_IOR(0x11, struct ethosu_uapi_buffer)
#define ETHOSU_IOCTL_BUFFER_GET ETHOSU_IOW(0x12, struct ethosu_uapi_buffer)
#define ETHOSU_IOCTL_NETWORK_CREATE ETHOSU_IOR(0x20, struct ethosu_uapi_network_create)
#define ETHOSU_IOCTL_NETWORK_INFO ETHOSU_IOR(0x21, struct ethosu_uapi_network_info)
#define ETHOSU_IOCTL_INFERENCE_CREATE \
  ETHOSU_IOR(0x30, struct ethosu_uapi_inference_create)
#define ETHOSU_IOCTL_INFERENCE_STATUS \
  ETHOSU_IOR(0x31, struct ethosu_uapi_result_status)
#define ETHOSU_IOCTL_INFERENCE_CANCEL \
  ETHOSU_IOR(0x32, struct ethosu_uapi_cancel_inference_status)
#define ETHOSU_IOCTL_INFERENCE_INVOKE \
  ETHOSU_IOR(0x33, struct ethosu_uapi_result_status)

#define ETHOSU_FD_MAX 16
#define ETHOSU_DIM_MAX 8
#define ETHOSU_PMU_EVENT_MAX 4

enum ethosu_uapi_status {
  ETHOSU_UAPI_STATUS_OK,
  ETHOSU_UAPI_STATUS_ERROR,
  ETHOSU_UAPI_STATUS_RUNNING,
  ETHOSU_UAPI_STATUS_REJECTED,
  ETHOSU_UAPI_STATUS_ABORTED,
  ETHOSU_UAPI_STATUS_ABORTING,
};

struct ethosu_uapi_buffer_create {
  __u32 capacity;
};

struct ethosu_uapi_buffer {
  __u32 offset;
  __u32 size;
};

enum ethosu_uapi_network_type {
  ETHOSU_UAPI_NETWORK_BUFFER = 1,
  ETHOSU_UAPI_NETWORK_INDEX,
};

struct ethosu_uapi_network_create {
  __u32 type;
  union {
    __u32 fd;
    __u32 index;
  };
};

struct ethosu_uapi_network_info {
  char desc[32];
  __u32 is_vela;
  __u32 ifm_count;
  __u32 ifm_size[ETHOSU_FD_MAX];
  __u32 ifm_types[ETHOSU_FD_MAX];
  __u32 ifm_offset[ETHOSU_FD_MAX];
  __u32 ifm_dims[ETHOSU_FD_MAX];
  __u32 ifm_shapes[ETHOSU_FD_MAX][ETHOSU_DIM_MAX];
  __u32 ofm_count;
  __u32 ofm_size[ETHOSU_FD_MAX];
  __u32 ofm_types[ETHOSU_FD_MAX];
  __u32 ofm_offset[ETHOSU_FD_MAX];
  __u32 ofm_dims[ETHOSU_FD_MAX];
  __u32 ofm_shapes[ETHOSU_FD_MAX][ETHOSU_DIM_MAX];
};

struct ethosu_uapi_pmu_config {
  __u32 events[ETHOSU_PMU_EVENT_MAX];
  __u32 cycle_count;
};

struct ethosu_uapi_pmu_counts {
  __u32 events[ETHOSU_PMU_EVENT_MAX];
  __u64 cycle_count;
};

struct ethosu_uapi_device_hw_id {
  __u32 version_status;
  __u32 version_minor;
  __u32 version_major;
  __u32 product_major;
  __u32 arch_patch_rev;
  __u32 arch_minor_rev;
  __u32 arch_major_rev;
};

struct ethosu_uapi_device_hw_cfg {
  __u32 macs_per_cc;
  __u32 cmd_stream_version;
  __u32 custom_dma;
};

struct ethosu_uapi_device_capabilities {
  struct ethosu_uapi_device_hw_id hw_id;
  struct ethosu_uapi_device_hw_cfg hw_cfg;
  __u32 driver_patch_rev;
  __u32 driver_minor_rev;
  __u32 driver_major_rev;
};

enum ethosu_uapi_inference_type {
  ETHOSU_UAPI_INFERENCE_MODEL = 0,
  ETHOSU_UAPI_INFERENCE_OP,
};

struct ethosu_uapi_memory_layout {
  __u32 flash_offset;
  __u32 arena_offset;
  __u32 input_count;
  __u32 input_offset[ETHOSU_FD_MAX];
  __u32 input_size[ETHOSU_FD_MAX];
  __u32 output_count;
  __u32 output_offset[ETHOSU_FD_MAX];
  __u32 output_size[ETHOSU_FD_MAX];
};

struct ethosu_uapi_inference_create {
  __u32 ifm_count;
  __u32 ifm_fd[ETHOSU_FD_MAX];
  __u32 ofm_count;
  __u32 ofm_fd[ETHOSU_FD_MAX];
  struct ethosu_uapi_memory_layout memory_layout;
  enum ethosu_uapi_inference_type inference_type;
  struct ethosu_uapi_pmu_config pmu_config;
};

struct ethosu_uapi_result_status {
  enum ethosu_uapi_status status;
  struct ethosu_uapi_pmu_config pmu_config;
  struct ethosu_uapi_pmu_counts pmu_count;
};

struct ethosu_uapi_cancel_inference_status {
  enum ethosu_uapi_status status;
};

#ifdef __cplusplus
} // namespace EthosU
#endif
