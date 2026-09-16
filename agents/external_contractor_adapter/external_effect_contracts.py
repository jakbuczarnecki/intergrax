# © Artur Czarnecki. All rights reserved.

"""External Work → ERL ``ExternalEffectContract`` declarations (GR-7-A2 / A2-R1)."""

from __future__ import annotations

from typing import Final

from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)

_RECONCILIATION_PROBES: Final = ("get_work",)


def _effective_support(
    *,
    operation_allows: bool,
    provider_supports: bool,
) -> ExternalEffectCapabilitySupport:
    if operation_allows and provider_supports:
        return ExternalEffectCapabilitySupport.SUPPORTED
    return ExternalEffectCapabilitySupport.NOT_SUPPORTED


def _base_contract(
    *,
    contract_id: str,
    operation_key: str,
    operation_idempotency_allowed: bool,
    capabilities: ExternalWorkProviderCapabilities,
) -> ExternalEffectContract:
    reconciliation = _effective_support(
        operation_allows=True,
        provider_supports=capabilities.supports_status_polling,
    )
    idempotency = _effective_support(
        operation_allows=operation_idempotency_allowed,
        provider_supports=capabilities.supports_idempotency,
    )
    probes = _RECONCILIATION_PROBES if reconciliation is ExternalEffectCapabilitySupport.SUPPORTED else ()
    return ExternalEffectContract(
        contract_id=contract_id,
        operation_key=operation_key,
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=idempotency,
            reconciliation=reconciliation,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=probes,
    )


def external_work_effect_contract_for_action(
    action: str,
    capabilities: ExternalWorkProviderCapabilities,
) -> ExternalEffectContract:
    """Effective ERL contract = canonical operation semantics ∩ provider capabilities."""
    if action == ACTION_CREATE_EXTERNAL_WORK:
        return _base_contract(
            contract_id="external_work.create_work.v1",
            operation_key="external_work.create_work",
            operation_idempotency_allowed=True,
            capabilities=capabilities,
        )
    if action == ACTION_ACCEPT_QUOTE:
        return _base_contract(
            contract_id="external_work.accept_quote.v1",
            operation_key="external_work.accept_quote",
            operation_idempotency_allowed=True,
            capabilities=capabilities,
        )
    if action == ACTION_CANCEL_EXTERNAL_WORK:
        return _base_contract(
            contract_id="external_work.cancel_work.v1",
            operation_key="external_work.cancel_work",
            operation_idempotency_allowed=False,
            capabilities=capabilities,
        )
    raise ValueError(f"unsupported external work action for ERL contract: {action!r}")


__all__ = [
    "external_work_effect_contract_for_action",
]
