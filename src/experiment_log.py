from __future__ import annotations

import csv
import fcntl
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch


GIB = 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def size_record(byte_count: int) -> dict[str, int | float]:
    return {"bytes": byte_count, "gib": round(byte_count / GIB, 4)}


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        with cpuinfo.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[-1].strip()
    return platform.processor() or "unknown"


def _nvidia_smi_devices() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        devices.append(
            {
                "index": int(row[0]),
                "full_model_name": row[1],
                "memory_total_mib": int(row[2]),
                "memory_total_gib": round(int(row[2]) / 1024, 2),
                "pci_bus_id": row[3],
                "driver_version": row[4],
            }
        )
    return devices


def _torch_cuda_devices() -> list[dict[str, Any]]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device: dict[str, Any] = {
            "logical_index": index,
            "full_model_name": properties.name,
            "total_memory": size_record(properties.total_memory),
        }
        pci_bus_id = getattr(properties, "pci_bus_id", None)
        if pci_bus_id is not None:
            device["pci_bus_id"] = pci_bus_id
        devices.append(device)
    return devices


def hardware_snapshot(world_size: int) -> dict[str, Any]:
    logical_cores = os.cpu_count() or 0
    physical_cores = psutil.cpu_count(logical=False) or 0
    memory = psutil.virtual_memory()

    if torch.cuda.is_available():
        device_type = "GPU"
        accelerator_backend = "CUDA"
        visible_gpu_count = torch.cuda.device_count()
    elif os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
        device_type = "TPU"
        accelerator_backend = "XLA"
        visible_gpu_count = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "GPU"
        accelerator_backend = "MPS"
        visible_gpu_count = 0
    else:
        device_type = "CPU"
        accelerator_backend = None
        visible_gpu_count = 0

    return {
        "captured_at_utc": utc_now(),
        "device_type": device_type,
        "accelerator_backend": accelerator_backend,
        "accelerator_count_used": world_size if device_type in {"GPU", "TPU"} else 0,
        "gpu_count_visible": visible_gpu_count,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_devices": _torch_cuda_devices() if torch.cuda.is_available() else [],
        "gpus": _nvidia_smi_devices() if torch.cuda.is_available() else [],
        "cpu": {
            "full_model_name": _cpu_model(),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
        },
        "system_ram": {
            "total": size_record(memory.total),
            "available_at_start": size_record(memory.available),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def local_accelerator_memory() -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None
    device_index = torch.cuda.current_device()
    torch.cuda.synchronize(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "device_index": device_index,
        "full_model_name": properties.name,
        "total_memory": size_record(properties.total_memory),
        "allocated_at_end": size_record(torch.cuda.memory_allocated(device_index)),
        "reserved_at_end": size_record(torch.cuda.memory_reserved(device_index)),
        "peak_allocated": size_record(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved": size_record(torch.cuda.max_memory_reserved(device_index)),
    }


def collect_accelerator_memory() -> list[dict[str, Any]]:
    local = local_accelerator_memory()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        return [item for item in gathered if item is not None]
    return [local] if local is not None else []


class ExperimentTracker:
    def __init__(
        self,
        *,
        experiment_type: str,
        formal_experiment: bool,
        platform_type: str,
        platform_name: str,
        log_root: Path,
        model_path: Path,
        data_root: Path,
        cache_dir: Path,
        output_dir: Path,
    ) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        default_id = f"{experiment_type}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
        self.run_id = os.environ.get("LATENTHALT_RUN_ID", default_id)
        self.log_root = Path(os.environ.get("LATENTHALT_LOG_ROOT", log_root)).resolve()
        self.run_dir = self.log_root / self.run_id
        self.experiment_type = experiment_type
        self.formal_experiment = formal_experiment
        self.platform_type = platform_type
        self.platform_name = platform_name
        self.model_path = model_path
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.started_at_unix = float(
            os.environ.get("LATENTHALT_RUN_STARTED_AT_UNIX", time.time())
        )
        self.started_at_utc = datetime.fromtimestamp(
            self.started_at_unix, timezone.utc
        ).isoformat()
        self._finished = False
        self.hardware = hardware_snapshot(self.world_size)

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            torch.cuda.reset_peak_memory_stats(local_rank)

        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(
                self.run_dir / "hardware.json",
                {
                    "run_id": self.run_id,
                    "experiment_type": self.experiment_type,
                    "formal_experiment": self.formal_experiment,
                    "platform": {
                        "type": self.platform_type,
                        "name": self.platform_name,
                    },
                    "hardware": self.hardware,
                    "storage_at_start": self._storage_snapshot(),
                },
            )

    def _storage_snapshot(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.output_dir.parent)
        model_bytes = directory_size(self.model_path)
        data_bytes = directory_size(self.data_root)
        cache_bytes = directory_size(self.cache_dir)
        output_bytes = directory_size(self.output_dir)
        run_log_bytes = directory_size(self.run_dir)
        checkpoints = []
        if self.output_dir.exists():
            for path in sorted(self.output_dir.glob("checkpoint-*")):
                if path.is_dir():
                    checkpoints.append(
                        {"path": str(path), "size": size_record(directory_size(path))}
                    )
        checkpoint_bytes = sum(item["size"]["bytes"] for item in checkpoints)
        observed_required = (
            model_bytes + data_bytes + cache_bytes + output_bytes + run_log_bytes
        )
        return {
            "source_model": size_record(model_bytes),
            "datasets": size_record(data_bytes),
            "intermediate_cache": size_record(cache_bytes),
            "experiment_output_total": size_record(output_bytes),
            "checkpoints_total": size_record(checkpoint_bytes),
            "checkpoints": checkpoints,
            "run_logs": size_record(run_log_bytes),
            "observed_required_total": size_record(observed_required),
            "filesystem": {
                "path": str(self.output_dir.parent),
                "total": size_record(disk.total),
                "used": size_record(disk.used),
                "free": size_record(disk.free),
            },
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)

    def _append_ledger_and_aggregate(self, record: dict[str, Any]) -> None:
        ledger_path = self.log_root / "experiments.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            records = []
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            formal = [
                item
                for item in records
                if item.get("formal_experiment") and item.get("status") == "completed"
            ]
            gpu_hours = sum(float(item.get("gpu_hours", 0.0)) for item in formal)
            accelerator_hours = sum(
                float(item.get("accelerator_hours", 0.0)) for item in formal
            )
            wall_clock = sum(float(item.get("wall_clock_seconds", 0.0)) for item in formal)
            aggregate = {
                "updated_at_utc": utc_now(),
                "scope": "all completed formal experiments in experiments.jsonl",
                "formal_experiments_completed": len(formal),
                "total_wall_clock_seconds_sum": round(wall_clock, 3),
                "total_wall_clock_hours_sum": round(wall_clock / 3600, 6),
                "total_gpu_hours": round(gpu_hours, 6),
                "total_accelerator_hours": round(accelerator_hours, 6),
                "run_ids": [item["run_id"] for item in formal],
            }
            self._write_json(self.log_root / "compute_summary.json", aggregate)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def finish(
        self,
        *,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        accelerator_memory: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        if self._finished or self.rank != 0:
            return
        self._finished = True
        ended_at_unix = time.time()
        wall_clock_seconds = max(0.0, ended_at_unix - self.started_at_unix)
        accelerator_count = int(self.hardware["accelerator_count_used"])
        gpu_count = accelerator_count if self.hardware["device_type"] == "GPU" else 0
        gpu_hours = gpu_count * wall_clock_seconds / 3600
        accelerator_hours = accelerator_count * wall_clock_seconds / 3600
        summary = {
            "run_id": self.run_id,
            "experiment_type": self.experiment_type,
            "formal_experiment": self.formal_experiment,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": datetime.fromtimestamp(
                ended_at_unix, timezone.utc
            ).isoformat(),
            "wall_clock_seconds": round(wall_clock_seconds, 3),
            "wall_clock_hours": round(wall_clock_seconds / 3600, 6),
            "gpu_count_used": gpu_count,
            "gpu_hours": round(gpu_hours, 6),
            "accelerator_count_used": accelerator_count,
            "accelerator_hours": round(accelerator_hours, 6),
            "platform": {"type": self.platform_type, "name": self.platform_name},
            "hardware": self.hardware,
            "accelerator_memory_usage": accelerator_memory or [],
            "storage_at_end": self._storage_snapshot(),
            "metrics": dict(metrics or {}),
            "error": error,
        }
        self._write_json(self.run_dir / "run_summary.json", summary)
        self._append_ledger_and_aggregate(summary)
