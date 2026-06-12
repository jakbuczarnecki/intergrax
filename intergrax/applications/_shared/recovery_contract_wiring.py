# © Artur Czarnecki. All rights reserved.

"""Validate and wire ApplicationRecoveryContract (APP-EVOL-5 · §49.5)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.application_recovery_contract import (
    APPLICATION_RECOVERY_CONTRACT_KEY,
    ApplicationRecoveryContract,
    CorruptCheckpointRecoveryAction,
    HostRestartRecoveryAction,
    TaskInterruptedRecoveryAction,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest

RECOVERY_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    "Runtime recovery",
    "Host restart",
    "Checkpoint",
    "In-flight tasks on deploy",
)


def validate_recovery_contract(
    reliability: ReliabilityProfile,
    *,
    strict_product: bool,
) -> list[str]:
    """Validate recovery contract consistency with reliability profile flags."""
    contract = reliability.recovery_contract
    if strict_product and contract is None:
        return ["recovery_contract required on STRICT product ReliabilityProfile"]

    violations: list[str] = []
    if contract is None:
        return violations

    if contract.on_host_restart is HostRestartRecoveryAction.RESUME_SCHEDULER:
        if not reliability.long_running_scheduler_enabled:
            violations.append(
                "on_host_restart=resume_scheduler requires long_running_scheduler_enabled",
            )

    if contract.on_task_interrupted is TaskInterruptedRecoveryAction.RESUME:
        if not reliability.idempotency_enabled:
            violations.append("on_task_interrupted=resume requires idempotency_enabled")
        if reliability.checkpoint_interval_steps < 1:
            violations.append(
                "on_task_interrupted=resume requires checkpoint_interval_steps >= 1",
            )

    if contract.on_corrupt_checkpoint is CorruptCheckpointRecoveryAction.REPLAY_FROM_SNAPSHOT:
        if strict_product and not contract.preserve_snapshot_id:
            violations.append(
                "replay_from_snapshot on STRICT product hosts requires preserve_snapshot_id",
            )

    return violations


def check_product_architecture_recovery_docs(
    package: str,
    *,
    applications_root: Path,
) -> list[str]:
    """Ensure product ARCHITECTURE.md documents recovery posture."""
    architecture_path = applications_root / package / "ARCHITECTURE.md"
    if not architecture_path.is_file():
        return [f"{package}: missing ARCHITECTURE.md for recovery documentation"]

    text = architecture_path.read_text(encoding="utf-8")
    violations: list[str] = []
    for marker in RECOVERY_ARCHITECTURE_MARKERS:
        if marker not in text:
            violations.append(f"{package}: ARCHITECTURE.md missing recovery marker {marker!r}")
    return violations


def validate_strict_product_recovery_contract(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    """Validate recovery contract and product architecture documentation."""
    if env.execution_mode is not ExecutionMode.STRICT:
        return []
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []

    strict_product = True
    violations = validate_recovery_contract(
        env.reliability_profile,
        strict_product=strict_product,
    )
    return violations


def check_strict_product_recovery_contract(
    product_id: str,
    manifest: ApplicationManifest,
    *,
    applications_root: Path,
) -> list[str]:
    """Return recovery-contract violations for one STRICT product host."""
    env = manifest.resolved_environment()
    package = f"{product_id}_application"

    violations = validate_strict_product_recovery_contract(manifest, env)
    violations.extend(
        check_product_architecture_recovery_docs(package, applications_root=applications_root),
    )
    prefix = f"{product_id}:"
    return [f"{prefix}{item}" for item in violations]


def attach_recovery_contract_to_task_metadata(
    task_metadata: dict[str, object],
    contract: ApplicationRecoveryContract | None,
) -> dict[str, object]:
    """Attach recovery contract wire payload to task metadata when declared."""
    if contract is None:
        return task_metadata
    updated = dict(task_metadata)
    updated[APPLICATION_RECOVERY_CONTRACT_KEY] = contract.model_dump(mode="json")
    return updated
