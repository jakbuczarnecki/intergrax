# © Artur Czarnecki. All rights reserved.

"""ERL Phase 3 — reconciliation probe execution contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidenceVerdict,
    ExternalEffectSafetyCapabilities,
    ReconciliationDisposition,
    ReconciliationExecutionError,
    ReconciliationProbeResult,
    UnknownUncertaintyPosture,
    build_reconciliation_plan,
    build_reconciliation_probe_request,
)

pytestmark = pytest.mark.unit


def _scheduled_plan():
    contract = ExternalEffectContract(
        contract_id="pay-1",
        operation_key="payments.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("payment_status",),
    )
    return build_reconciliation_plan(
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        contract=contract,
        plugin_id="reconcile-pay",
        probe_ref="payment_status",
    )


def test_build_probe_request_from_scheduled_plan() -> None:
    plan = _scheduled_plan()
    request = build_reconciliation_probe_request(
        plan=plan,
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="pay-1",
        attempt_index=2,
    )
    assert request.probe_ref == "payment_status"
    assert request.plugin_id == "reconcile-pay"
    assert request.attempt_index == 2


def test_build_probe_request_rejects_non_scheduled_plan() -> None:
    contract = ExternalEffectContract(
        contract_id="x",
        operation_key="op",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    plan = build_reconciliation_plan(
        unknown_posture=UnknownUncertaintyPosture.ESCALATE_REQUIRED,
        contract=contract,
        plugin_id="unused",
        probe_ref="payment_status",
    )
    assert plan.disposition is ReconciliationDisposition.ESCALATE_REQUIRED
    with pytest.raises(ReconciliationExecutionError, match="schedule_probe"):
        build_reconciliation_probe_request(
            plan=plan,
            tenant_id="t",
            correlation_id="c",
            contract_id="x",
        )


def test_probe_result_requires_evidence_ref() -> None:
    with pytest.raises(ValueError):
        ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="",
        )
