"""
Shared helpers for the runtime-measurement scripts (time_feature_computation.py)
and train.py's timing instrumentation.

Timer wraps time.perf_counter() with a torch.cuda.synchronize() before and
after the timed region whenever `device` is a CUDA device, so asynchronous GPU
kernels are fully counted (and not attributed to whichever region happens to
synchronize next).
"""

import csv
import json
import os
import platform
import sys
import time
from typing import Dict, Iterable, List, Optional

import torch

SUMMARY_HEADER = ["dataset", "step", "n_items", "seconds", "mean_per_item_s", "device"]


def _is_cuda(device) -> bool:
    if device is None:
        return False
    return torch.cuda.is_available() and getattr(device, "type", str(device)).startswith("cuda")


def sync(device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize()


def device_str(device) -> str:
    return "cpu" if device is None else str(device)


class Timer:
    """`with Timer(device) as t: ...` -> t.seconds (float) after the block."""

    def __init__(self, device=None):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0
        return False


def summary_row(dataset: str, step: str, seconds: float, n_items: Optional[int], device) -> Dict:
    """One row of the shared train-side summary CSV (SUMMARY_HEADER)."""
    mean = (seconds / n_items) if n_items else ""
    return {
        "dataset": dataset,
        "step": step,
        "n_items": n_items if n_items is not None else "",
        "seconds": f"{seconds:.6f}",
        "mean_per_item_s": f"{mean:.6f}" if mean != "" else "",
        "device": device_str(device),
    }


def append_rows_csv(path: str, header: List[str], rows: Iterable[Dict]) -> None:
    """Append dict rows to `path`, writing the header only if the file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def env_info(device=None) -> Dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": device_str(device),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_env_json(path: str, device=None, extra: Optional[Dict] = None) -> None:
    payload = env_info(device)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
