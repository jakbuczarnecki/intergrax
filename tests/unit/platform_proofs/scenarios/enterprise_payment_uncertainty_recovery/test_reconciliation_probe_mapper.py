# © Artur Czarnecki. All rights reserved.

"""Mapper tests — variant dataset drives verdicts without branch-per-variant code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.reconciliation_execution import ReconciliationProbeRequest
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    EXTERNAL_EFFECT_SOR_PROBE_REF,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.probe_result import (
    map_snapshot_to_probe_result,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    resolve_sor_truth_fields,
)

pytestmark = pytest.mark.unit

_SCENARIO_ROOT = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
)
_VARIANT_IDS = (
    "payment_completed_after_unknown",
    "payment_failed_after_unknown",
    "payment_truth_unavailable",
)


def _snapshot_for_variant(variant_id: str, correlation_id: str = "corr-mapper") -> ExternalRealitySnapshot:
    variant_path = _SCENARIO_ROOT / "dataset/variants" / variant_id / "scenario_variant.json"
    document = json.loads(variant_path.read_text(encoding="utf-8"))
    fields = resolve_sor_truth_fields(document)
    return ExternalRealitySnapshot(
        correlation_id=correlation_id,
        external_effect_reference="EXT-MAPPER",
        terminal_outcome=fields.terminal_outcome,
        funds_captured=fields.funds_captured,
        truth_availability_state=fields.truth_availability_state,
        sor_transaction_ref="SOR-MAPPER",
    )


def _request(correlation_id: str = "corr-mapper") -> ReconciliationProbeRequest:
    return ReconciliationProbeRequest(
        tenant_id="tenant-lab",
        correlation_id=correlation_id,
        contract_id="contract-lab",
        probe_ref=EXTERNAL_EFFECT_SOR_PROBE_REF,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        attempt_index=1,
    )


@pytest.mark.parametrize(
    ("variant_id", "expected_verdict"),
    [
        ("payment_completed_after_unknown", ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS),
        ("payment_failed_after_unknown", ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE),
        ("payment_truth_unavailable", ExternalEffectEvidenceVerdict.INSUFFICIENT),
    ],
    ids=_VARIANT_IDS,
)
def test_variant_dataset_maps_to_platform_verdict(
    variant_id: str,
    expected_verdict: ExternalEffectEvidenceVerdict,
) -> None:
    result = map_snapshot_to_probe_result(
        request=_request(),
        snapshot=_snapshot_for_variant(variant_id),
    )
    assert result.verdict is expected_verdict
    assert result.evidence_ref.startswith("evidence://erl-qual-004/reconcile/corr-mapper/1/")
    assert result.rationale


def test_invalid_probe_ref_is_insufficient() -> None:
    request = ReconciliationProbeRequest(
        tenant_id="tenant-lab",
        correlation_id="corr-mapper",
        contract_id="contract-lab",
        probe_ref="unsupported.probe",
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
    )
    result = map_snapshot_to_probe_result(
        request=request,
        snapshot=_snapshot_for_variant("payment_completed_after_unknown"),
    )
    assert result.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT
    assert "unsupported_probe_ref" in result.rationale


def test_inconsistent_sor_state_is_insufficient() -> None:
    snapshot = ExternalRealitySnapshot(
        correlation_id="corr-mapper",
        external_effect_reference="EXT-MAPPER",
        terminal_outcome="PAYMENT_COMPLETED",
        funds_captured=False,
        truth_availability_state="AVAILABLE",
        sor_transaction_ref="SOR-MAPPER",
    )
    result = map_snapshot_to_probe_result(request=_request(), snapshot=snapshot)
    assert result.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT
    assert "funds_captured" in result.rationale
