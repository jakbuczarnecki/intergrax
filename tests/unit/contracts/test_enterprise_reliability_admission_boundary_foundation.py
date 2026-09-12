# © Artur Czarnecki. All rights reserved.

"""ERL admission boundary contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectAdmissionContextError,
    ExternalEffectAdmissionRequest,
    ExternalEffectAdmissionSourceContext,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    assert_external_effect_admission_request,
    uncertainty_state_ref_for_correlation,
)

pytestmark = pytest.mark.unit


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


def _request(**updates: object) -> ExternalEffectAdmissionRequest:
    base = ExternalEffectAdmissionRequest(
        external_effect_ref="ext:effect:corr-a",
        effect_outcome=ExternalEffectOutcome.UNKNOWN,
        correlation_id="corr-a",
        contract=_contract(),
        source_context=ExternalEffectAdmissionSourceContext(
            source_kind="integration",
            source_ref="exec:run:1",
        ),
    )
    return base.model_copy(update=updates)


def test_assert_admission_request_accepts_valid_unknown_context() -> None:
    assert_external_effect_admission_request(_request())


def test_assert_admission_request_rejects_mismatched_uncertainty_ref() -> None:
    with pytest.raises(ExternalEffectAdmissionContextError, match="uncertainty_state_ref"):
        assert_external_effect_admission_request(
            _request(uncertainty_state_ref="erl:uncertainty:other"),
        )


def test_assert_admission_request_rejects_blank_correlation() -> None:
    with pytest.raises(ExternalEffectAdmissionContextError, match="correlation_id"):
        assert_external_effect_admission_request(_request(correlation_id="   "))


def test_uncertainty_state_ref_is_correlation_stable() -> None:
    assert uncertainty_state_ref_for_correlation("corr-x") == "erl:uncertainty:corr-x"
