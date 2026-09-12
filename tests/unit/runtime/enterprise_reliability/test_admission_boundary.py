# © Artur Czarnecki. All rights reserved.

"""ERL admission boundary runtime tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectAdmissionContextError,
    ExternalEffectAdmissionPhase,
    ExternalEffectAdmissionRequest,
    ExternalEffectAdmissionSourceContext,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectReliabilityInteraction,
    ExternalEffectSafetyCapabilities,
    ReliabilityCaseLifecycleState,
    UnknownUncertaintyPosture,
    uncertainty_state_ref_for_correlation,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_into_enterprise_reliability,
)

pytestmark = pytest.mark.unit

_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "admission_boundary.py"
)


def _contract() -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="capture-1",
        operation_key="provider.capture",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("provider_status",),
    )


def test_admit_unknown_creates_case_and_preserves_correlation() -> None:
    request = ExternalEffectAdmissionRequest(
        external_effect_ref="ext:payment:intent-9",
        effect_outcome=ExternalEffectOutcome.UNKNOWN,
        correlation_id="corr-9",
        contract=_contract(),
        source_context=ExternalEffectAdmissionSourceContext(
            source_kind="integration",
            source_ref="exec:child:9",
        ),
    )
    result = admit_external_effect_into_enterprise_reliability(request)
    assert result.phase is ExternalEffectAdmissionPhase.HANDED_OFF
    assert result.correlation_id == "corr-9"
    assert result.external_effect_ref == "ext:payment:intent-9"
    assert (
        result.projection.interaction
        is ExternalEffectReliabilityInteraction.UNCERTAINTY_FAIL_CLOSED
    )
    assert result.uncertainty_state is not None
    assert result.uncertainty_state.correlation_id == "corr-9"
    assert result.unknown_posture is UnknownUncertaintyPosture.RECONCILE_ONLY
    assert result.case_record is not None
    assert result.case_record.correlation_id == "corr-9"
    assert result.case_record.lifecycle_state is (
        ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING
    )
    assert result.case_record.refs.uncertainty_state_ref == uncertainty_state_ref_for_correlation(
        "corr-9",
    )


def test_admit_definitive_success_skips_case_creation() -> None:
    request = ExternalEffectAdmissionRequest(
        external_effect_ref="ext:effect:ok",
        effect_outcome=ExternalEffectOutcome.SUCCESS,
        correlation_id="corr-ok",
        contract=_contract(),
        source_context=ExternalEffectAdmissionSourceContext(
            source_kind="integration",
            source_ref="exec:1",
        ),
    )
    result = admit_external_effect_into_enterprise_reliability(request)
    assert result.case_record is None
    assert result.uncertainty_state is None
    assert (
        result.projection.interaction
        is ExternalEffectReliabilityInteraction.NO_FAILURE_CLASSIFICATION
    )


def test_admit_rejects_invalid_context() -> None:
    request = ExternalEffectAdmissionRequest(
        external_effect_ref="ext:effect:bad",
        effect_outcome=ExternalEffectOutcome.UNKNOWN,
        correlation_id="corr-bad",
        contract=_contract(),
        source_context=ExternalEffectAdmissionSourceContext(
            source_kind="integration",
            source_ref="exec:1",
        ),
        uncertainty_state_ref="erl:uncertainty:wrong",
    )
    with pytest.raises(ExternalEffectAdmissionContextError):
        admit_external_effect_into_enterprise_reliability(request)


def test_admission_boundary_does_not_run_downstream_capabilities() -> None:
    source = _RUNTIME_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "plan_external_effect_reconciliation",
        "execute_external_effect_reconciliation_probe",
        "plan_external_effect_resolution",
        "execute_external_effect_compensation",
        "EnterpriseReliabilityPluginGateway",
        "InMemoryEnterpriseReliabilityPluginRegistry",
    )
    for token in forbidden:
        assert token not in source
