# © Artur Czarnecki. All rights reserved.

"""GR-7-A2-R1 — effective ExternalEffectContract from provider capabilities."""

from __future__ import annotations

import pytest

from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    evaluate_unknown_uncertainty_posture,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)

pytestmark = pytest.mark.unit


def _caps(**updates: object) -> ExternalWorkProviderCapabilities:
    base = ExternalWorkProviderCapabilities(
        provider_id="test-provider",
        supports_create=True,
        supports_accept=True,
        supports_cancel=True,
        supports_status_polling=True,
        supports_idempotency=True,
    )
    return base.model_copy(update=updates)


def test_create_idempotency_supported_when_provider_allows() -> None:
    contract = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        _caps(supports_idempotency=True),
    )
    assert contract.safety.idempotency is ExternalEffectCapabilitySupport.SUPPORTED


def test_create_idempotency_not_supported_when_provider_disallows() -> None:
    contract = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        _caps(supports_idempotency=False),
    )
    assert contract.safety.idempotency is ExternalEffectCapabilitySupport.NOT_SUPPORTED


def test_accept_idempotency_follows_provider_capability() -> None:
    supported = external_work_effect_contract_for_action(
        ACTION_ACCEPT_QUOTE,
        _caps(supports_idempotency=True),
    )
    unsupported = external_work_effect_contract_for_action(
        ACTION_ACCEPT_QUOTE,
        _caps(supports_idempotency=False),
    )
    assert supported.safety.idempotency is ExternalEffectCapabilitySupport.SUPPORTED
    assert unsupported.safety.idempotency is ExternalEffectCapabilitySupport.NOT_SUPPORTED


def test_cancel_idempotency_never_supported_even_if_provider_idempotent() -> None:
    contract = external_work_effect_contract_for_action(
        ACTION_CANCEL_EXTERNAL_WORK,
        _caps(supports_idempotency=True),
    )
    assert contract.safety.idempotency is ExternalEffectCapabilitySupport.NOT_SUPPORTED


def test_reconciliation_requires_status_polling() -> None:
    supported = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        _caps(supports_status_polling=True),
    )
    unsupported = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        _caps(supports_status_polling=False),
    )
    assert supported.safety.reconciliation is ExternalEffectCapabilitySupport.SUPPORTED
    assert unsupported.safety.reconciliation is ExternalEffectCapabilitySupport.NOT_SUPPORTED
    assert unsupported.reconciliation_probe_refs == ()


def test_no_idempotency_provider_cannot_get_reconcile_or_idempotent_repeat() -> None:
    contract = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        _caps(supports_idempotency=False, supports_status_polling=False),
    )
    assert (
        evaluate_unknown_uncertainty_posture(contract)
        is UnknownUncertaintyPosture.ESCALATE_REQUIRED
    )
