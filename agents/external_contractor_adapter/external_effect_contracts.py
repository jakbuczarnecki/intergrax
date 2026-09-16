# © Artur Czarnecki. All rights reserved.

"""External Work → ERL ``ExternalEffectContract`` declarations (GR-7-A2)."""

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

_CREATE_CONTRACT: Final = ExternalEffectContract(
    contract_id="external_work.create_work.v1",
    operation_key="external_work.create_work",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
    reconciliation_probe_refs=("get_work",),
)

_ACCEPT_CONTRACT: Final = ExternalEffectContract(
    contract_id="external_work.accept_quote.v1",
    operation_key="external_work.accept_quote",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
    reconciliation_probe_refs=("get_work",),
)

_CANCEL_CONTRACT: Final = ExternalEffectContract(
    contract_id="external_work.cancel_work.v1",
    operation_key="external_work.cancel_work",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
    reconciliation_probe_refs=("get_work",),
)

_ACTION_CONTRACTS: Final = {
    ACTION_CREATE_EXTERNAL_WORK: _CREATE_CONTRACT,
    ACTION_ACCEPT_QUOTE: _ACCEPT_CONTRACT,
    ACTION_CANCEL_EXTERNAL_WORK: _CANCEL_CONTRACT,
}


def external_work_effect_contract_for_action(action: str) -> ExternalEffectContract:
    """Resolve the ERL effect contract for a canonical external-work side-effect action."""
    contract = _ACTION_CONTRACTS.get(action)
    if contract is None:
        raise ValueError(f"unsupported external work action for ERL contract: {action!r}")
    return contract


__all__ = [
    "external_work_effect_contract_for_action",
]
