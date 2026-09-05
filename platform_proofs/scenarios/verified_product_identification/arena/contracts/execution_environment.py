"""Typed arena execution environment contracts — provider-neutral."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArenaAcceleratorRequirement(str, Enum):
    """Hardware accelerator requirement declared by an arena execution profile."""

    ANY = "ANY"
    CUDA = "CUDA"


class ArenaExecutionEnvironmentStatus(str, Enum):
    """Typed classification for arena pre-flight environment validation."""

    READY = "READY"
    FAILED_EXECUTION_ENVIRONMENT = "FAILED_EXECUTION_ENVIRONMENT"
    BLOCKED_CUDA_RUNTIME_ENVIRONMENT = "BLOCKED_CUDA_RUNTIME_ENVIRONMENT"


@dataclass(frozen=True, slots=True)
class ArenaExecutionEnvironmentSnapshot:
    """Captured execution environment evidence before candidate work begins."""

    profile_id: str
    accelerator_requirement: ArenaAcceleratorRequirement
    requested_device: str | None
    resolved_device: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_count: int
    total_vram_bytes: int | None
    torch_version: str | None
    cuda_runtime_version: str | None
    python_executable: str
    status: ArenaExecutionEnvironmentStatus
    detail: str | None
