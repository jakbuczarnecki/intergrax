# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trusted ExternalEffectContract declarations for TL8 redrive (G-A02 — not caller-supplied)."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCategory,
    ExternalEffectCapabilitySupport,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
)

TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT: Final = ExternalEffectContract(
    contract_id="tool_life.platform_durable_physical_lifecycle.v1",
    operation_key="platform.external_operation.physical_lifecycle",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
    reconciliation_probe_refs=("tool_life.external_operation_state_store.load",),
)

TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT: Final = ExternalEffectContract(
    contract_id="tool_life.http_idempotent_post.qual.v1",
    operation_key="http_rest.idempotent_post",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
)

TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT: Final = ExternalEffectContract(
    contract_id="tool_life.http_blind_post.qual.v1",
    operation_key="http_rest.blind_post",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
)

TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT: Final = ExternalEffectContract(
    contract_id="tool_life.mcp_tools_call.default.qual.v1",
    operation_key="mcp_catalog.tools_call",
    category=ExternalEffectCategory.INFRASTRUCTURE,
    safety=ExternalEffectSafetyCapabilities(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    ),
)

_TRUSTED_REDRIVE_EFFECT_CONTRACTS: Final = (
    TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT,
)

_TRUSTED_BY_CONTRACT_ID: Final = {
    contract.contract_id: contract for contract in _TRUSTED_REDRIVE_EFFECT_CONTRACTS
}


def resolve_trusted_redrive_effect_contract(contract_id: str) -> ExternalEffectContract:
    """Authoritative registry lookup — callers cannot invent new contract_ids."""
    from intergrax.contracts.external_operation_redrive_authority import (
        ExternalOperationRedriveEffectClassRejectedError,
    )

    normalized = contract_id.strip()
    trusted = _TRUSTED_BY_CONTRACT_ID.get(normalized)
    if trusted is None:
        raise ExternalOperationRedriveEffectClassRejectedError(
            f"untrusted redrive effect contract_id {contract_id!r}",
        )
    return trusted


__all__ = [
    "TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT",
    "TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT",
    "TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT",
    "TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT",
    "resolve_trusted_redrive_effect_contract",
]
