"""Architecture tests for ERL-QUAL-004 scenario data provisioning boundary."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
    ProvisioningExecutionContext,
    ScenarioIdentity,
    ScenarioVariantSelection,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.failures import (
    ProvisioningFailure,
    ProvisioningFailureCode,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.lifecycle import (
    ProvisioningLifecycleCoordinator,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
    ProvisioningPhase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.results import (
    CleanupPhaseOutcome,
    ProvisioningPhaseOutcome,
    ProvisioningPreparationHandle,
    ProvisioningSessionReference,
    StateAvailabilityOutcome,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.in_memory import (
    InMemoryReferenceProvisioner,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
_DATASET_ROOT = _SCENARIO_ROOT / "dataset"
_CONTRACTS_ROOT = _SCENARIO_ROOT / "contracts/provisioning"

_FORBIDDEN_VENDOR_ROOTS = frozenset(
    {
        "sqlite3",
        "psycopg",
        "asyncpg",
        "sqlalchemy",
        "boto3",
        "redis",
        "pymongo",
        "requests",
        "httpx",
    }
)


def _erl_context(*, variant_id: str, run_id: str = "test-run") -> ProvisioningContext:
    return ProvisioningContext(
        identity=ScenarioIdentity(
            qualification_id="ERL-QUAL-004",
            scenario_slug="enterprise_payment_uncertainty_recovery",
        ),
        variant=ScenarioVariantSelection(variant_id=variant_id),
        execution=ProvisioningExecutionContext(
            run_id=run_id,
            dataset_package_root=_DATASET_ROOT,
        ),
    )


def _manifest_variant_ids() -> tuple[str, ...]:
    manifest = json.loads((_DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    return tuple(entry["variant_id"] for entry in variants)


def _imported_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".", 1)[0].casefold())
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module.split(".", 1)[0].casefold())
    return modules


@dataclass
class SequenceTrackingProvisioner:
    """Independent contract implementation for lifecycle-order tests."""

    calls: list[str] = field(default_factory=list)
    fail_prepare: bool = False
    fail_cleanup: bool = False

    def prepare(self, context: ProvisioningContext) -> ProvisioningPhaseOutcome:
        self.calls.append("prepare")
        if self.fail_prepare:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INVALID_DATASET,
                    phase=ProvisioningPhase.PREPARE,
                    message="forced prepare failure",
                ),
            )
        preparation = ProvisioningPreparationHandle(
            qualification_id=context.identity.qualification_id,
            variant_id=context.variant.variant_id,
            logical_fingerprint="tracking-fingerprint",
        )
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PREPARE,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            preparation=preparation,
        )

    def provision(
        self,
        context: ProvisioningContext,
        preparation: ProvisioningPreparationHandle,
    ) -> ProvisioningPhaseOutcome:
        self.calls.append("provision")
        session = ProvisioningSessionReference(
            session_id="tracking-session",
            variant_id=preparation.variant_id,
            logical_fingerprint=preparation.logical_fingerprint,
            execution_handles=("tracking-handle",),
        )
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PROVISION,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
        )

    def state_availability(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> StateAvailabilityOutcome:
        self.calls.append("state_availability")
        return StateAvailabilityOutcome(
            phase=ProvisioningPhase.STATE_AVAILABILITY,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
            state_ready=True,
        )

    def cleanup(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> CleanupPhaseOutcome:
        self.calls.append("cleanup")
        if self.fail_cleanup:
            return CleanupPhaseOutcome(
                phase=ProvisioningPhase.CLEANUP,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.CLEANUP_FAILURE,
                    phase=ProvisioningPhase.CLEANUP,
                    message="forced cleanup failure",
                ),
            )
        return CleanupPhaseOutcome(
            phase=ProvisioningPhase.CLEANUP,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
        )


@dataclass
class MinimalHashProvisioner:
    """Second independent implementation — proves replaceability without shared base class."""

    label: str

    def prepare(self, context: ProvisioningContext) -> ProvisioningPhaseOutcome:
        preparation = ProvisioningPreparationHandle(
            qualification_id=context.identity.qualification_id,
            variant_id=context.variant.variant_id,
            logical_fingerprint=f"{self.label}-prep",
        )
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PREPARE,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            preparation=preparation,
        )

    def provision(
        self,
        context: ProvisioningContext,
        preparation: ProvisioningPreparationHandle,
    ) -> ProvisioningPhaseOutcome:
        session = ProvisioningSessionReference(
            session_id=f"{self.label}-session",
            variant_id=preparation.variant_id,
            logical_fingerprint=preparation.logical_fingerprint,
            execution_handles=(self.label,),
        )
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PROVISION,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
        )

    def state_availability(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> StateAvailabilityOutcome:
        return StateAvailabilityOutcome(
            phase=ProvisioningPhase.STATE_AVAILABILITY,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
            state_ready=True,
        )

    def cleanup(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> CleanupPhaseOutcome:
        return CleanupPhaseOutcome(
            phase=ProvisioningPhase.CLEANUP,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
        )


def test_provisioning_contract_implemented_independently() -> None:
    coordinator = ProvisioningLifecycleCoordinator()
    for impl in (
        SequenceTrackingProvisioner(),
        MinimalHashProvisioner(label="minimal-a"),
        MinimalHashProvisioner(label="minimal-b"),
    ):
        result = coordinator.run(impl, _erl_context(variant_id="payment_completed_after_unknown"))
        assert result.status is ProvisioningOutcomeStatus.SUCCEEDED
        assert result.session is not None
        assert result.session.execution_handles


def test_lifecycle_order_is_respected() -> None:
    port = SequenceTrackingProvisioner()
    coordinator = ProvisioningLifecycleCoordinator()
    result = coordinator.run(port, _erl_context(variant_id="payment_completed_after_unknown"))
    assert result.status is ProvisioningOutcomeStatus.SUCCEEDED
    assert port.calls == ["prepare", "provision", "state_availability", "cleanup"]

    failed_port = SequenceTrackingProvisioner(fail_prepare=True)
    failed = coordinator.run(failed_port, _erl_context(variant_id="payment_completed_after_unknown"))
    assert failed.status is ProvisioningOutcomeStatus.FAILED
    assert failed_port.calls == ["prepare"]


def test_failures_are_explicit_not_exception_only() -> None:
    context = _erl_context(variant_id="does-not-exist")
    outcome = InMemoryReferenceProvisioner().prepare(context)
    assert outcome.status is ProvisioningOutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code is ProvisioningFailureCode.MISSING_SCENARIO_VARIANT

    unavailable = InMemoryReferenceProvisioner(unavailable=True).prepare(
        _erl_context(variant_id="payment_completed_after_unknown")
    )
    assert unavailable.failure is not None
    assert unavailable.failure.code is ProvisioningFailureCode.PROVISIONING_UNAVAILABLE

    coordinator = ProvisioningLifecycleCoordinator()
    cleanup_fail = coordinator.run(
        SequenceTrackingProvisioner(fail_cleanup=True),
        _erl_context(variant_id="payment_completed_after_unknown"),
    )
    assert cleanup_fail.status is ProvisioningOutcomeStatus.FAILED
    assert cleanup_fail.cleanup is not None
    assert cleanup_fail.cleanup.failure is not None
    assert cleanup_fail.cleanup.failure.code is ProvisioningFailureCode.CLEANUP_FAILURE


def test_provisioning_contracts_have_no_vendor_imports() -> None:
    violations: list[str] = []
    for module_path in sorted(_CONTRACTS_ROOT.rglob("*.py")):
        forbidden = sorted(_FORBIDDEN_VENDOR_ROOTS & _imported_roots(module_path))
        if forbidden:
            rel = module_path.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}: {', '.join(forbidden)}")
    assert violations == []


@pytest.mark.parametrize("variant_id", _manifest_variant_ids())
def test_variant_selection_without_changing_provisioner_logic(variant_id: str) -> None:
    provisioner = InMemoryReferenceProvisioner()
    coordinator = ProvisioningLifecycleCoordinator()
    result = coordinator.run(provisioner, _erl_context(variant_id=variant_id, run_id=variant_id))
    assert result.status is ProvisioningOutcomeStatus.SUCCEEDED
    assert result.session is not None
    assert result.session.variant_id == variant_id


def test_reference_provisioner_runs_full_lifecycle_on_canonical_dataset() -> None:
    result = ProvisioningLifecycleCoordinator().run(
        InMemoryReferenceProvisioner(),
        _erl_context(variant_id="payment_truth_unavailable"),
    )
    assert result.status is ProvisioningOutcomeStatus.SUCCEEDED
    assert result.state_availability is not None
    assert result.state_availability.state_ready is True
