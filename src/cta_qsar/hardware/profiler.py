"""Hardware detection: CPU/RAM/GPU available to the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class HardwareProfile:
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_memory_gb: float | None
    cuda_available: bool
    torch_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "gpu_available": self.gpu_available,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cuda_available": self.cuda_available,
            "torch_available": self.torch_available,
        }

    @property
    def compute_tier(self) -> str:
        if self.gpu_available and self.cuda_available:
            return "gpu"
        return "cpu"


def probe() -> HardwareProfile:
    """Probe the current machine."""
    try:
        import os as _os

        cores = _os.cpu_count() or 1
    except Exception:  # pragma: no cover
        cores = 1

    try:
        import psutil  # type: ignore[import-not-found]

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = _estimate_ram_gb()

    gpu_available = False
    gpu_memory_gb: float | None = None
    cuda_available = False

    try:
        import torch  # type: ignore[import-not-found]

        torch_available = True
        if torch.cuda.is_available():
            gpu_available = True
            cuda_available = True
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                gpu_memory_gb = None
    except ImportError:
        torch_available = False

    return HardwareProfile(
        cpu_cores=cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_memory_gb=gpu_memory_gb,
        cuda_available=cuda_available,
        torch_available=torch_available,
    )


def _estimate_ram_gb() -> float:
    """Best-effort RAM estimate without psutil."""
    try:
        result = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        return float(result)
    except (AttributeError, ValueError):  # pragma: no cover
        return 8.0