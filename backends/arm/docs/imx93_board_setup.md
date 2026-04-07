# i.MX93 Board Setup

Board-side changes used to expose Ethos-U and run ExecuTorch against the BSP stack.

## Boot / DTB

- Copied the Ethos-U DTB from rootfs `/boot` to the real boot partition.
- Backed up the active boot-partition `grub.cfg`.
- Restored this line in the real boot GRUB config:

```text
devicetree (${root})/boot/${fdtfile}
```

Purpose: boot with the expected DTB so the BSP exposes Ethos-U correctly.

## Remoteproc / CM33

- Enabled `remoteproc-cm33`.
- Relevant runtime path:

```text
/sys/class/remoteproc/remoteproc0/
```

- Observed firmware name: `ethosu_firmware`

Useful debug reset:

```bash
echo stop > /sys/class/remoteproc/remoteproc0/state
```

This avoided stale `running` state issues during testing.

## Board Build Environment

Used these board-local fixes before rebuilding ExecuTorch:

```bash
ln -s /root/executorch-support/torchgen/packaged /packaged
export PYTHONPATH=/root/executorch:/root/executorch/src:/root/executorch-support
cmake --build /root/executorch/cmake-out-linux-ethosu-bsp -j4
```

Result:

```text
/root/executorch/cmake-out-linux-ethosu-bsp/executor_runner
```

## Network / Access

- Restored Ethernet with a static link-local address.
- Switched to SSH/SCP for file transfer.
- USB serial console was also used for recovery and debugging.

Representative commands:

```bash
ssh root@169.254.15.28
scp <model>.pte root@169.254.15.28:/root/titok/
```

## Board-Only Debug Patch

- A temporary debug-only patch was used at:

```text
/root/ethos-u-linux-driver-stack/driver_library/src/ethosu.cpp
```

- It relaxed the `0.0.0` driver-version check.
- This was not a required permanent setup step.

## What These Changes Proved

- The earlier userspace/kernel ABI mismatch was fixed.
- `/dev/ethosu0` and the BSP Ethos-U stack were usable.
- BSP Ethos-U example flows could run after a clean `remoteproc` restart.
- The remaining issue is in the ExecuTorch invocation path, not basic board bring-up.
