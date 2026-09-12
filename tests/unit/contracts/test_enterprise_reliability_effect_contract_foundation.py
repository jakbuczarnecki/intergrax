# © Artur Czarnecki. All rights reserved.

"""ERL Phase 2 — external effect contract foundation tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
    contract_declares_idempotency,
    evaluate_unknown_uncertainty_posture,
    project_external_effect_to_reliability,
)
from intergrax.contracts.execution_retry import ExecutionFailureKind

pytestmark = pytest.mark.unit


def _contract(
    *,
    idempotency: ExternalEffectCapabilitySupport,
    reconciliation: ExternalEffectCapabilitySupport,
    compensation: ExternalEffectCapabilitySupport = ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    reconciliation_probe_refs: tuple[str, ...] = (),
    compensation_operation_ref: str | None = None,
) -> ExternalEffectContract:
    probes = reconciliation_probe_refs
    if (
        reconciliation is ExternalEffectCapabilitySupport.SUPPORTED
        and not probes
    ):
        probes = ("status_probe",)
    return ExternalEffectContract(
        contract_id="contract-1",
        operation_key="integration.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=idempotency,
            reconciliation=reconciliation,
            compensation=compensation,
        ),
        reconciliation_probe_refs=probes,
        compensation_operation_ref=compensation_operation_ref,
    )


def test_reconciliation_requires_probe_refs() -> None:
    with pytest.raises(ValueError, match="reconciliation_probe_refs required"):
        ExternalEffectContract(
            contract_id="c",
            operation_key="op",
            category=ExternalEffectCategory.INVENTORY,
            safety=ExternalEffectSafetyCapabilities(
                idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
                reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
                compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            ),
        )


def test_compensation_requires_operation_ref() -> None:
    with pytest.raises(ValueError, match="compensation_operation_ref required"):
        ExternalEffectContract(
            contract_id="c",
            operation_key="op",
            category=ExternalEffectCategory.FINANCIAL,
            safety=ExternalEffectSafetyCapabilities(
                idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
                reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
                compensation=ExternalEffectCapabilitySupport.SUPPORTED,
            ),
        )


def test_unknown_posture_prefers_idempotent_declaration() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    assert (
        evaluate_unknown_uncertainty_posture(contract)
        is UnknownUncertaintyPosture.RECONCILE_OR_IDEMPOTENT_REPEAT
    )


def test_unknown_posture_reconcile_only_without_idempotency() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    assert (
        evaluate_unknown_uncertainty_posture(contract)
        is UnknownUncertaintyPosture.RECONCILE_ONLY
    )


def test_unknown_posture_escalate_when_no_safe_paths() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    )
    assert (
        evaluate_unknown_uncertainty_posture(contract)
        is UnknownUncertaintyPosture.ESCALATE_REQUIRED
    )


def test_contract_drives_reliability_unknown_side_effect_flag() -> None:
    idempotent = _contract(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    )
    projection = project_external_effect_to_reliability(
        ExternalEffectOutcome.UNKNOWN,
        effect_contract=idempotent,
        side_effect_idempotency_guaranteed=False,
    )
    assert projection.classification is not None
    assert projection.classification.kind is ExecutionFailureKind.UNKNOWN
    assert projection.classification.has_unknown_side_effect is False
    assert contract_declares_idempotency(idempotent)

    non_idempotent = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    risky = project_external_effect_to_reliability(
        ExternalEffectOutcome.UNKNOWN,
        effect_contract=non_idempotent,
        side_effect_idempotency_guaranteed=True,
    )
    assert risky.classification is not None
    assert risky.classification.has_unknown_side_effect is True
