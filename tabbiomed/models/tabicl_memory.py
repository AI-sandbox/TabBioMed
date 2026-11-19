"""Job-aware memory limits for TabICL's existing inference/offloading manager."""

from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import psutil
import torch
from tabicl.model.inference import InferenceManager, OffloadMode

MIB = 1024**2


def cgroup_memory_limits(
    cgroup_file=Path("/proc/self/cgroup"), mount_file=Path("/proc/self/mountinfo")
):
    """Return (limit, usage) for every applicable v1/v2 ancestor, including job parents."""
    try:
        memberships = [
            line.split(":", 2) for line in cgroup_file.read_text().splitlines()
        ]
        mounts = mount_file.read_text().splitlines()
    except OSError:
        return []
    pairs = []
    for line in mounts:
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        v2 = fs[0] == "cgroup2"
        if not v2 and "memory" not in fs[2].split(","):
            continue
        unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
        mount_root = PurePosixPath(unescape(fields[3]))
        mount = Path(unescape(fields[4]))
        for _, controllers, member in memberships:
            if (v2 and controllers) or (
                not v2 and "memory" not in controllers.split(",")
            ):
                continue
            try:
                relative = PurePosixPath(member).relative_to(mount_root)
            except ValueError:
                if member == "/":
                    relative = PurePosixPath(".")
                else:
                    continue
            current = mount / relative
            while True:
                try:
                    limit = int(
                        (
                            current / ("memory.max" if v2 else "memory.limit_in_bytes")
                        ).read_text()
                    )
                    usage = int(
                        (
                            current
                            / ("memory.current" if v2 else "memory.usage_in_bytes")
                        ).read_text()
                    )
                    if 0 < limit < 2**60:
                        pairs.append((limit, usage))
                except (OSError, ValueError):
                    pass
                if current == mount:
                    break
                current = current.parent
    return pairs


def _env_int(name, default=0):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def available_job_memory():
    """Available bytes bounded by host, cgroup, and Slurm limits."""
    available = psutil.virtual_memory().available
    limits = cgroup_memory_limits()
    for limit, usage in limits:
        available = min(available, max(0, limit - usage))
    slurm_mb = _env_int("SLURM_MEM_PER_NODE", 0)
    if not slurm_mb:
        slurm_mb = _env_int("SLURM_MEM_PER_CPU", 0) * _env_int(
            "SLURM_CPUS_PER_TASK", 1
        )
    if slurm_mb:
        smallest = min(limits, default=None)
        usage = (
            smallest[1] if smallest and smallest[0] <= slurm_mb * MIB * 1.05 else None
        )
        if usage is None:
            process = psutil.Process()
            usage = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    usage += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        available = min(available, max(0, slurm_mb * MIB - usage))
    return available


_LOCAL_FSTYPES = ("ext4", "ext3", "xfs", "btrfs", "zfs", "overlay", "apfs")


def local_offload_root():
    """Find real local disk, never shared storage or RAM-backed tmpfs."""
    mounts = psutil.disk_partitions(all=True)
    candidates = [
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        "/tmp",
        "/data/tmp",
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).resolve()
        if not path.is_dir() or not os.access(path, os.W_OK):
            continue
        matches = [m for m in mounts if path.is_relative_to(Path(m.mountpoint))]
        if not matches:
            continue
        mount = max(
            matches,
            key=lambda m: (len(m.mountpoint), m.fstype in _LOCAL_FSTYPES),
        )
        if mount.fstype not in _LOCAL_FSTYPES:
            continue
        if psutil.disk_usage(path).free > 1024 * MIB:
            return path
    raise RuntimeError(
        "TabICL requires writable node-local disk for temporary offloading"
    )


class JobMemoryInferenceManager(InferenceManager):
    def get_available_cpu_memory(self):
        return available_job_memory() / MIB

    def _resolve_offload_mode(self, output_mb, gpu_free_mb, cpu_free_mb, disk_free_mb):
        mode, reason = super()._resolve_offload_mode(
            output_mb, gpu_free_mb, cpu_free_mb, disk_free_mb
        )
        if mode == OffloadMode.CPU and output_mb > cpu_free_mb * self.cpu_safety_factor:
            raise MemoryError(
                f"TabICL output needs {output_mb / 1024:.1f} GiB, job RAM and local disk "
                "cannot safely hold it. Refusing the upstream CPU/swap fallback."
            )
        return mode, reason

    def _allocate_output_buffer(self, mode, shape, dtype):
        size = self._estimate_tensor_mb(shape, dtype)
        if mode == OffloadMode.GPU:
            try:
                return torch.empty(shape, dtype=dtype, device=self.exe_device), {
                    "mode": mode.name
                }
            except torch.cuda.OutOfMemoryError:
                mode = OffloadMode.CPU
        if (
            mode == OffloadMode.CPU
            and size > self.get_available_cpu_memory() * self.cpu_safety_factor
        ):
            disk_free = (
                self.get_available_disk_space(self.disk_offload_dir)
                if self.disk_offload_dir
                else 0.0
            )
            if (
                self.disk_offload_dir
                and size
                <= max(0, disk_free - self.disk_min_free_mb) * self.disk_safety_factor
            ):
                mode = OffloadMode.DISK
            else:
                raise MemoryError(
                    "TabICL output no longer fits the remaining job RAM or local disk"
                )
        return super()._allocate_output_buffer(mode, shape, dtype)


@contextmanager
def bounded_inference(estimator):
    """Configure per-model managers, with per-call temporary files and guaranteed cleanup."""
    core = getattr(estimator, "model_", None)
    if (
        core is None
        or getattr(getattr(estimator, "device_", None), "type", None) != "cuda"
    ):
        yield
        return
    configs = [
        estimator.inference_config_.COL_CONFIG,
        estimator.inference_config_.ROW_CONFIG,
        estimator.inference_config_.ICL_CONFIG,
    ]
    keys = dict(
        cpu_safety_factor=0.75,
        disk_min_free_mb=1024.0,
        disk_flush_mb=512.0,
        disk_cleanup=True,
        max_pinned_memory_mb=1024.0,
        async_depth=1,
    )
    missing = object()
    saved = [
        {key: getattr(config, key, missing) for key in (*keys, "disk_offload_dir")}
        for config in configs
    ]
    managers = []
    temp_dir = None
    try:
        for module in core.modules():
            original = getattr(module, "inference_mgr", None)
            if isinstance(original, InferenceManager):
                manager = JobMemoryInferenceManager(
                    original.enc_name, original.out_dim, original.out_no_seq
                )
                managers.append((module, original, manager))
                module.inference_mgr = manager
        try:
            offload_root = local_offload_root()
        except RuntimeError:
            offload_root = None
        if offload_root is not None:
            temp_dir = tempfile.TemporaryDirectory(prefix="tabicl_", dir=offload_root)
            directory = temp_dir.name
        else:
            directory = None
        for config in configs:
            for key, value in {**keys, "disk_offload_dir": directory}.items():
                setattr(config, key, value)
        yield
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        for config, previous in zip(configs, saved):
            for key, value in previous.items():
                if value is missing:
                    if hasattr(config, key):
                        delattr(config, key)
                else:
                    setattr(config, key, value)
        for module, original, manager in managers:
            if manager._buffer_pool is not None:
                manager._buffer_pool.clear()
            module.inference_mgr = original
