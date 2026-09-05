"""GPU gate tests for embedding arena execution environment validation."""

from __future__ import annotations

import pytest

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.execution_profiles import (
    SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
    STANDARD_ARENA_EXECUTION_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    ArenaExecutionEnvironmentError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_environment import (
    ArenaAcceleratorRequirement,
    ArenaExecutionEnvironmentStatus,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.execution_environment import (
    probe_arena_execution_environment,
    validate_arena_execution_environment,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.runner import (
    run_embedding_arena,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    HardwareRuntimeCapability,
)

pytestmark = pytest.mark.unit


def _fake_hardware(
    *,
    torch_version: str | None = "2.14.0+cpu",
    cuda_available: bool = False,
    gpu_count: int = 0,
    gpu_name: str | None = None,
    total_vram_bytes: int | None = None,
    cuda_runtime_version: str | None = None,
) -> HardwareRuntimeCapability:
    return HardwareRuntimeCapability(
        python_version="3.12.0",
        platform="test",
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_runtime_version=cuda_runtime_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_bytes=total_vram_bytes,
        sentence_transformers_version=None,
        configured_device=None,
        resolved_provider_device=None,
        provider_device_proof="unavailable",
    )


def _configuration(device: str | None) -> VpiEmbeddingProviderExecutionConfiguration:
    return VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(device=device)
    )


def _patch_hardware(
    monkeypatch: pytest.MonkeyPatch,
    hardware: HardwareRuntimeCapability,
) -> None:
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.execution_environment.probe_hardware_runtime_capability",
        lambda **_: hardware,
    )


def test_safe_local_gpu_cuda_unavailable_fails_before_runner_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(torch_version="2.14.0+cpu", cuda_available=False),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.runner._load_arena_context",
        lambda **_: (_ for _ in ()).throw(AssertionError("arena context must not load")),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.runner.run_candidate_phase_subprocess",
        lambda **_: (_ for _ in ()).throw(AssertionError("subprocess must not spawn")),
    )

    configuration = _configuration("cuda")
    with pytest.raises(ArenaExecutionEnvironmentError) as exc_info:
        validate_arena_execution_environment(
            SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
            execution_configuration=configuration,
        )

    assert (
        exc_info.value.snapshot.status
        is ArenaExecutionEnvironmentStatus.BLOCKED_CUDA_RUNTIME_ENVIRONMENT
    )

    with pytest.raises(ArenaExecutionEnvironmentError):
        run_embedding_arena(
            execution_budget=SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
            session_dir=".tmp/session/test-gpu-gate",
        )


def test_safe_local_gpu_resolved_cpu_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(torch_version="2.14.0", cuda_available=True, gpu_count=1),
    )
    configuration = _configuration("cpu")

    with pytest.raises(ArenaExecutionEnvironmentError) as exc_info:
        validate_arena_execution_environment(
            SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
            execution_configuration=configuration,
        )

    assert (
        exc_info.value.snapshot.status
        is ArenaExecutionEnvironmentStatus.FAILED_EXECUTION_ENVIRONMENT
    )
    assert exc_info.value.snapshot.resolved_device == "cpu"


def test_safe_local_gpu_cuda_available_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(
            torch_version="2.14.0+cu124",
            cuda_available=True,
            gpu_count=1,
            gpu_name="NVIDIA GeForce RTX 4080 Laptop GPU",
            total_vram_bytes=12_282_000_000,
            cuda_runtime_version="12.4",
        ),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.execution_environment.assert_execution_device_available",
        lambda configuration: None,
    )

    snapshot = validate_arena_execution_environment(
        SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
        execution_configuration=_configuration("cuda"),
    )

    assert snapshot.status is ArenaExecutionEnvironmentStatus.READY
    assert snapshot.resolved_device == "cuda"
    assert snapshot.gpu_name == "NVIDIA GeForce RTX 4080 Laptop GPU"


def test_standard_profile_cpu_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(torch_version="2.14.0+cpu", cuda_available=False),
    )

    snapshot = validate_arena_execution_environment(
        STANDARD_ARENA_EXECUTION_BUDGET,
        execution_configuration=_configuration(None),
    )

    assert snapshot.status is ArenaExecutionEnvironmentStatus.READY


def test_candidate_child_validates_same_cuda_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(torch_version="2.14.0+cpu", cuda_available=False),
    )
    calls: list[str] = []

    def _record_validate(execution_budget, **kwargs) -> None:
        calls.append(execution_budget.profile_id)
        raise ArenaExecutionEnvironmentError(
            probe_arena_execution_environment(
                execution_budget,
                execution_configuration=_configuration("cuda"),
            )
        )

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.run_embedding_arena_candidate.validate_arena_execution_environment",
        _record_validate,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.run_embedding_arena_candidate.execute_candidate_stage_ab",
        lambda **_: (_ for _ in ()).throw(AssertionError("candidate work must not run")),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.run_embedding_arena_candidate.load_proof_environment",
        lambda **_: None,
    )

    from platform_proofs.scenarios.verified_product_identification.run_embedding_arena_candidate import (
        main,
    )

    exit_code = main(
        [
            "--candidate-id",
            "bge-m3",
            "--profile",
            "safe-local-gpu",
            "--phase",
            "stage_ab",
            "--session-dir",
            ".tmp/session/test-gpu-gate-child",
        ]
    )

    assert exit_code == 1
    assert calls == ["safe-local-gpu"]


def test_no_candidate_subprocess_on_failed_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hardware(
        monkeypatch,
        _fake_hardware(torch_version="2.14.0+cpu", cuda_available=False),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.runner._load_arena_context",
        lambda **_: (
            (),
            None,
            None,
            None,
            None,
        ),
    )
    spawn_called = False

    def _spawn(**kwargs) -> None:
        nonlocal spawn_called
        spawn_called = True

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.runner.run_candidate_phase_subprocess",
        _spawn,
    )

    with pytest.raises(ArenaExecutionEnvironmentError):
        run_embedding_arena(
            execution_budget=SAFE_LOCAL_GPU_MICRO_ARENA_EXECUTION_BUDGET,
            session_dir=".tmp/session/test-gpu-gate-no-spawn",
        )

    assert spawn_called is False


def test_invalid_accelerator_requirement_rejected() -> None:
    with pytest.raises(ValueError):
        ArenaAcceleratorRequirement("TPU")
