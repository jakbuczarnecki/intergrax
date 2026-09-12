# © Artur Czarnecki. All rights reserved.

"""ERL Phase 3 — reconciliation framework foundation contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
    ReconciliationDisposition,
    ReconciliationPlanningError,
    UnknownUncertaintyPosture,
    assert_reconciliation_probe_ref,
    build_reconciliation_plan,
    default_contract_probe_ref,
    default_reconciliation_bounds,
    evaluate_reconciliation_disposition,
)

pytestmark = pytest.mark.unit


def _contract(
    *,
    reconciliation: ExternalEffectCapabilitySupport,
    probes: tuple[str, ...] = ("status_probe",),
) -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="pay-1",
        operation_key="payments.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=reconciliation,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=probes
        if reconciliation is ExternalEffectCapabilitySupport.SUPPORTED
        else (),
    )


def test_disposition_escalate_when_posture_requires() -> None:
    contract = _contract(reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED)
    assert (
        evaluate_reconciliation_disposition(
            unknown_posture=UnknownUncertaintyPosture.ESCALATE_REQUIRED,
            contract=contract,
        )
        is ReconciliationDisposition.ESCALATE_REQUIRED
    )


def test_disposition_no_probe_for_idempotent_only_contract() -> None:
    contract = ExternalEffectContract(
        contract_id="idem-1",
        operation_key="inventory.reserve",
        category=ExternalEffectCategory.INVENTORY,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    assert (
        evaluate_reconciliation_disposition(
            unknown_posture=UnknownUncertaintyPosture.RECONCILE_OR_IDEMPOTENT_REPEAT,
            contract=contract,
        )
        is ReconciliationDisposition.NO_DECLARED_PROBE
    )


def test_assert_probe_ref_must_be_declared_on_contract() -> None:
    contract = _contract(reconciliation=ExternalEffectCapabilitySupport.SUPPORTED)
    assert_reconciliation_probe_ref(contract, "status_probe")
    with pytest.raises(ReconciliationPlanningError, match="not declared"):
        assert_reconciliation_probe_ref(contract, "other_probe")


def test_build_plan_schedules_declared_probe_with_bounds() -> None:
    contract = _contract(reconciliation=ExternalEffectCapabilitySupport.SUPPORTED)
    plan = build_reconciliation_plan(
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        contract=contract,
        plugin_id="stripe-reconcile",
        probe_ref=default_contract_probe_ref(contract),
    )
    assert plan.disposition is ReconciliationDisposition.SCHEDULE_PROBE
    assert plan.probe_ref == "status_probe"
    assert plan.plugin_id == "stripe-reconcile"
    assert plan.bounds == default_reconciliation_bounds()


def test_build_plan_without_schedule_probe_omits_probe_fields() -> None:
    contract = _contract(reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED)
    plan = build_reconciliation_plan(
        unknown_posture=UnknownUncertaintyPosture.ESCALATE_REQUIRED,
        contract=contract,
        plugin_id="unused",
        probe_ref="status_probe",
    )
    assert plan.disposition is ReconciliationDisposition.ESCALATE_REQUIRED
    assert plan.probe_ref is None
