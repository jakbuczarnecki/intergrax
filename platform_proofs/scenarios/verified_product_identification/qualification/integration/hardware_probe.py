"""Runtime hardware capability probe — optional torch imports only."""

from __future__ import annotations

import platform
import sys

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    HardwareRuntimeCapability,
)


def _optional_import_versions() -> tuple[str | None, str | None]:
    torch_version: str | None = None
    sentence_transformers_version: str | None = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        pass
    try:
        import sentence_transformers

        sentence_transformers_version = sentence_transformers.__version__
    except ImportError:
        pass
    return torch_version, sentence_transformers_version


def probe_hardware_runtime_capability(
    *,
    configured_device: str | None,
    resolved_provider_device: str | None = None,
    provider_device_proof: str = "unavailable",
) -> HardwareRuntimeCapability:
    torch_version, sentence_transformers_version = _optional_import_versions()
    cuda_available = False
    cuda_runtime_version: str | None = None
    gpu_name: str | None = None
    gpu_count = 0
    total_vram_bytes: int | None = None

    if torch_version is not None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                properties = torch.cuda.get_device_properties(0)
                total_vram_bytes = int(properties.total_memory)
            cuda_runtime_version = torch.version.cuda

    return HardwareRuntimeCapability(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_runtime_version=cuda_runtime_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_bytes=total_vram_bytes,
        sentence_transformers_version=sentence_transformers_version,
        configured_device=configured_device,
        resolved_provider_device=resolved_provider_device,
        provider_device_proof=provider_device_proof,
    )
