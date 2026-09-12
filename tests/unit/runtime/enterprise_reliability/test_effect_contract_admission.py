# © Artur Czarnecki. All rights reserved.

"""ERL Phase 2 — contract-aware UNKNOWN admission runtime tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown_with_contract,
)

pytestmark = pytest.mark.unit


def test_admit_unknown_with_contract_attaches_posture() -> None:
    contract = ExternalEffectContract(
        contract_id="ship-label",
        operation_key="carrier.create_label",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("carrier_label_status",),
    )
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-ship",
        contract=contract,
    )
    assert admission.contract_id == "ship-label"
    assert admission.unknown_posture is UnknownUncertaintyPosture.RECONCILE_ONLY
    assert admission.state.effect_outcome is ExternalEffectOutcome.UNKNOWN
