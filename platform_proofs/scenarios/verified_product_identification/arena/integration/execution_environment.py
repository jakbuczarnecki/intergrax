"""Arena execution environment validation — integration/runtime composition."""

from __future__ import annotations

import sys

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingDeviceUnavailableError,
    VpiEmbeddingProviderExecutionConfiguration,
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    ArenaExecutionEnvironmentError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_budget import (
    EmbeddingArenaExecutionBudget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_environment import (
    ArenaAcceleratorRequirement,
    ArenaExecutionEnvironmentSnapshot,
    ArenaExecutionEnvironmentStatus,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    HardwareRuntimeCapability,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.hardware_probe import (
    probe_hardware_runtime_capability,
)


def resolve_accelerator_device(device: str | None) -> str | None:
    """Normalize operator device strings to a canonical accelerator label."""
    if device is None:
        return None
    normalized = device.strip().casefold()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    return device.strip()


def _build_snapshot(
    *,
    execution_budget: EmbeddingArenaExecutionBudget,
    execution_configuration: VpiEmbeddingProviderExecutionConfiguration,
    hardware: HardwareRuntimeCapability,
    status: ArenaExecutionEnvironmentStatus,
    detail: str | None,
) -> ArenaExecutionEnvironmentSnapshot:
    requested_device = execution_configuration.device
    return ArenaExecutionEnvironmentSnapshot(
        profile_id=execution_budget.profile_id,
        accelerator_requirement=execution_budget.accelerator_requirement,
        requested_device=requested_device,
        resolved_device=resolve_accelerator_device(requested_device),
        cuda_available=hardware.cuda_available,
        gpu_name=hardware.gpu_name,
        gpu_count=hardware.gpu_count,
        total_vram_bytes=hardware.total_vram_bytes,
        torch_version=hardware.torch_version,
        cuda_runtime_version=hardware.cuda_runtime_version,
        python_executable=sys.executable,
        status=status,
        detail=detail,
    )


def _classify_cuda_requirement(
    *,
    execution_budget: EmbeddingArenaExecutionBudget,
    execution_configuration: VpiEmbeddingProviderExecutionConfiguration,
    hardware: HardwareRuntimeCapability,
) -> ArenaExecutionEnvironmentSnapshot:
    resolved_device = resolve_accelerator_device(execution_configuration.device)
    if hardware.torch_version is None:
        return _build_snapshot(
            execution_budget=execution_budget,
            execution_configuration=execution_configuration,
            hardware=hardware,
            status=ArenaExecutionEnvironmentStatus.BLOCKED_CUDA_RUNTIME_ENVIRONMENT,
            detail="CUDA accelerator required but torch is not installed",
        )
    if resolved_device is None:
        return _build_snapshot(
            execution_budget=execution_budget,
            execution_configuration=execution_configuration,
            hardware=hardware,
            status=ArenaExecutionEnvironmentStatus.FAILED_EXECUTION_ENVIRONMENT,
            detail=(
                "CUDA accelerator required but VPI_EMBEDDING_DEVICE is unset; "
                "set VPI_EMBEDDING_DEVICE=cuda"
            ),
        )
    if resolved_device != "cuda":
        return _build_snapshot(
            execution_budget=execution_budget,
            execution_configuration=execution_configuration,
            hardware=hardware,
            status=ArenaExecutionEnvironmentStatus.FAILED_EXECUTION_ENVIRONMENT,
            detail=(
                "CUDA accelerator required but resolved device is "
                f"{resolved_device!r}; set VPI_EMBEDDING_DEVICE=cuda"
            ),
        )
    try:
        assert_execution_device_available(execution_configuration)
    except VpiEmbeddingDeviceUnavailableError as exc:
        return _build_snapshot(
            execution_budget=execution_budget,
            execution_configuration=execution_configuration,
            hardware=hardware,
            status=ArenaExecutionEnvironmentStatus.BLOCKED_CUDA_RUNTIME_ENVIRONMENT,
            detail=str(exc),
        )
    if not hardware.cuda_available or hardware.gpu_count < 1:
        return _build_snapshot(
            execution_budget=execution_budget,
            execution_configuration=execution_configuration,
            hardware=hardware,
            status=ArenaExecutionEnvironmentStatus.BLOCKED_CUDA_RUNTIME_ENVIRONMENT,
            detail="CUDA accelerator required but no GPU is available in the current torch build",
        )
    return _build_snapshot(
        execution_budget=execution_budget,
        execution_configuration=execution_configuration,
        hardware=hardware,
        status=ArenaExecutionEnvironmentStatus.READY,
        detail=None,
    )


def probe_arena_execution_environment(
    execution_budget: EmbeddingArenaExecutionBudget,
    *,
    execution_configuration: VpiEmbeddingProviderExecutionConfiguration | None = None,
) -> ArenaExecutionEnvironmentSnapshot:
    configuration = (
        execution_configuration
        if execution_configuration is not None
        else load_vpi_embedding_provider_execution_configuration()
    )
    hardware = probe_hardware_runtime_capability(
        configured_device=configuration.device,
    )
    if execution_budget.accelerator_requirement is ArenaAcceleratorRequirement.CUDA:
        return _classify_cuda_requirement(
            execution_budget=execution_budget,
            execution_configuration=configuration,
            hardware=hardware,
        )
    return _build_snapshot(
        execution_budget=execution_budget,
        execution_configuration=configuration,
        hardware=hardware,
        status=ArenaExecutionEnvironmentStatus.READY,
        detail=None,
    )


def validate_arena_execution_environment(
    execution_budget: EmbeddingArenaExecutionBudget,
    *,
    execution_configuration: VpiEmbeddingProviderExecutionConfiguration | None = None,
) -> ArenaExecutionEnvironmentSnapshot:
    snapshot = probe_arena_execution_environment(
        execution_budget,
        execution_configuration=execution_configuration,
    )
    if snapshot.status is not ArenaExecutionEnvironmentStatus.READY:
        raise ArenaExecutionEnvironmentError(snapshot)
    return snapshot
