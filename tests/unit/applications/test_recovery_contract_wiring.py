# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-5 — ApplicationRecoveryContract validation and task wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications._shared.recovery_contract_wiring import (
    check_strict_product_recovery_contract,
    validate_recovery_contract,
)
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.application_recovery_contract import (
    APPLICATION_RECOVERY_CONTRACT_KEY,
    ApplicationRecoveryContract,
    HostRestartRecoveryAction,
    TaskInterruptedRecoveryAction,
    standard_strict_product_recovery_contract,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_APPLICATIONS = Path(__file__).resolve().parents[3] / "applications"


def test_product_defaults_include_recovery_contract() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="demo.product")
    assert env.reliability_profile.recovery_contract is not None
    assert env.reliability_profile.recovery_contract.preserve_snapshot_id is True


def test_validate_recovery_contract_requires_scheduler_for_resume() -> None:
    reliability = ReliabilityProfile(
        long_running_scheduler_enabled=False,
        recovery_contract=ApplicationRecoveryContract(
            on_host_restart=HostRestartRecoveryAction.RESUME_SCHEDULER,
        ),
    )
    violations = validate_recovery_contract(reliability, strict_product=True)
    assert any("resume_scheduler" in item for item in violations)


def test_validate_recovery_contract_resume_requires_idempotency() -> None:
    reliability = ReliabilityProfile(
        idempotency_enabled=False,
        recovery_contract=ApplicationRecoveryContract(
            on_task_interrupted=TaskInterruptedRecoveryAction.RESUME,
        ),
    )
    violations = validate_recovery_contract(reliability, strict_product=True)
    assert any("idempotency_enabled" in item for item in violations)


def test_apply_reliability_task_defaults_attaches_recovery_contract() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="demo.product")
    task = Task(
        task_id="task-1",
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="demo.cap"),
    )
    updated = apply_reliability_task_defaults(task, env)
    payload = updated.metadata[APPLICATION_RECOVERY_CONTRACT_KEY]
    assert payload["on_host_restart"] == standard_strict_product_recovery_contract().on_host_restart.value


@pytest.mark.parametrize("product_id,manifest", list(iter_strict_product_manifests()))
def test_reference_strict_product_hosts_pass_recovery_gate(
    product_id: str,
    manifest: object,
) -> None:
    violations = check_strict_product_recovery_contract(
        product_id,
        manifest,  # type: ignore[arg-type]
        applications_root=REPO_APPLICATIONS,
    )
    assert violations == []
