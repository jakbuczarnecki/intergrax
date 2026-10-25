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
from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)

_CATALOG_TOOLS_REQUIRING_QUAL_PROVENANCE: Final = frozenset(
    {
        "http.idempotent_post.qual",
    },
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

_CONTRACT_IDS_REQUIRING_COMPLETE_PHYSICAL_IDENTITY: Final = frozenset(
    {
        TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
    },
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

TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT: Final = ExternalEffectContract(
    contract_id="tool_life.http_read_only.qual.v1",
    operation_key="http_rest.read_only",
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
    TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT,
)

_TRUSTED_BY_CONTRACT_ID: Final = {
    contract.contract_id: contract for contract in _TRUSTED_REDRIVE_EFFECT_CONTRACTS
}

# Sanctioned provider/tool → trusted contract (qual registry; not caller-invented).
_QUAL_SANCTIONED_EXECUTION_BINDING: Final = {
    ("http_rest", "http_idempotent_post"): TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
    ("http_rest", "http_blind_post"): TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
    ("http_rest", "http_read_only"): TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT.contract_id,
    (
        "platform",
        "external_operation_physical_lifecycle",
    ): TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT.contract_id,
    (
        "mcp_catalog",
        "tools_call",
    ): TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT.contract_id,
}

# Authoritative catalog tool_id → (contract_id, provider_id, provider_execution_tool_id).
# No permissive default: unknown catalog tools must not inherit platform lifecycle Class B.
_CATALOG_TOOL_SANCTIONED_REDRIVE_BINDING: Final = {
    "http.idempotent_post.qual": (
        TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
        "http_rest",
        "http_idempotent_post",
    ),
    "http_blind_post": (
        TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
        "http_rest",
        "http_blind_post",
    ),
    "http.read_only": (
        TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT.contract_id,
        "http_rest",
        "http_read_only",
    ),
    "external_operation_physical_lifecycle": (
        TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT.contract_id,
        "platform",
        "external_operation_physical_lifecycle",
    ),
    "invoke_catalog_tool": (
        TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT.contract_id,
        "mcp_catalog",
        "tools_call",
    ),
}


def assert_trusted_redrive_binding_matches_execution_context(
    contract: ExternalEffectContract,
    *,
    provider_id: str | None,
    tool_id: str | None,
) -> None:
    """Reject binding without sanctioned provider/tool execution identity."""
    if provider_id is None or tool_id is None:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "trusted redrive binding requires sanctioned provider_id and tool_id "
            "execution identity",
        )
    key = (provider_id.strip(), tool_id.strip())
    if not key[0] or not key[1]:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "trusted redrive binding requires sanctioned provider_id and tool_id "
            "execution identity",
        )
    expected_contract_id = _QUAL_SANCTIONED_EXECUTION_BINDING.get(key)
    if expected_contract_id is None:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            f"no sanctioned redrive effect binding for provider/tool {key!r}",
        )
    if contract.contract_id != expected_contract_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "trusted effect contract does not match sanctioned provider/tool context",
        )


def trusted_redrive_contract_requires_complete_physical_identity(contract_id: str) -> bool:
    """HTTP idempotency-backed Class B requires bound physical key and effect fingerprint."""
    return contract_id.strip() in _CONTRACT_IDS_REQUIRING_COMPLETE_PHYSICAL_IDENTITY


def assert_sanctioned_class_b_catalog_tool_provenance(
    *,
    catalog_tool_id: str,
    registered_tool: object | None = None,
) -> None:
    """Reject catalog-ID impersonation without trusted handler/activation provenance."""
    from intergrax.tools.providers.http.class_b_qualification_provenance import (
        registered_tool_matches_sanctioned_http_idempotent_post_qual,
    )
    from intergrax.tools.providers.http.idempotent_post_qual import (
        HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
    )
    from intergrax.tools.registry.runtime import RegisteredTool

    normalized = catalog_tool_id.strip()
    if normalized not in _CATALOG_TOOLS_REQUIRING_QUAL_PROVENANCE:
        return
    if normalized == HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID:
        if not isinstance(registered_tool, RegisteredTool):
            raise ExternalOperationRedriveEffectBindingMismatchError(
                "catalog tool lacks sanctioned Class B qualification registration",
            )
        if not registered_tool_matches_sanctioned_http_idempotent_post_qual(
            registered_tool,
        ):
            raise ExternalOperationRedriveEffectBindingMismatchError(
                "catalog tool handler or activation does not match sanctioned "
                "Class B qualification registration",
            )
        return
    raise ExternalOperationRedriveEffectBindingMismatchError(
        f"no sanctioned Class B provenance policy for catalog tool {normalized!r}",
    )


def require_complete_physical_effect_identity_values(
    *,
    physical_idempotency_key: str | None,
    physical_effect_fingerprint: str | None,
) -> None:
    if not (physical_idempotency_key and physical_idempotency_key.strip()):
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "Class B physical idempotency key is required",
        )
    if not (physical_effect_fingerprint and physical_effect_fingerprint.strip()):
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "Class B physical effect fingerprint is required",
        )


def catalog_tool_has_sanctioned_redrive_effect_binding(catalog_tool_id: str) -> bool:
    """True when catalog tool_id has an explicit trusted effect mapping (not a default)."""
    normalized_tool = catalog_tool_id.strip()
    return bool(normalized_tool) and normalized_tool in _CATALOG_TOOL_SANCTIONED_REDRIVE_BINDING


def resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
    catalog_tool_id: str,
) -> tuple[str, str, str]:
    """
    Resolve authoritative (contract_id, provider_id, binding_tool_id) for durable binding.

    Requires an explicit catalog entry; unknown tools cannot acquire platform lifecycle Class B.
    """
    normalized_tool = catalog_tool_id.strip()
    if not normalized_tool:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "catalog tool_id is required for sanctioned redrive effect binding",
        )
    resolved = _CATALOG_TOOL_SANCTIONED_REDRIVE_BINDING.get(normalized_tool)
    if resolved is None:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            f"no sanctioned redrive effect binding for catalog tool {normalized_tool!r}",
        )
    return resolved


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
    "TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT",
    "TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT",
    "TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT",
    "assert_sanctioned_class_b_catalog_tool_provenance",
    "assert_trusted_redrive_binding_matches_execution_context",
    "catalog_tool_has_sanctioned_redrive_effect_binding",
    "require_complete_physical_effect_identity_values",
    "resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool",
    "resolve_trusted_redrive_effect_contract",
    "trusted_redrive_contract_requires_complete_physical_identity",
]
