# © Artur Czarnecki. All rights reserved.

"""Configured marketplace tool execution target correlation (TRACE-X-P5-R2-P3-R2-R1)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.tools.marketplace_tool_execution_routing import (
    _CONFIGURED_TARGET_PREFIX,
    derive_marketplace_configured_tool_execution_intent_target_correlation,
    derive_marketplace_configured_tool_execution_target_reference,
)

_BINDING_A = (
    "configured-capability-binding:"
    "configured-capability-execution:recovery-1:decision-1:"
    "configured-capability-execution:recovery-1:decision-1:fp-1"
)
_BINDING_B = (
    "configured-capability-binding:"
    "configured-capability-execution:recovery-2:decision-2:"
    "configured-capability-execution:recovery-2:decision-2:fp-2"
)


def test_configured_target_reference_matches_intent_correlation_derivation() -> None:
    assert derive_marketplace_configured_tool_execution_target_reference(
        _BINDING_A,
    ) == derive_marketplace_configured_tool_execution_intent_target_correlation(_BINDING_A)


def test_configured_target_reference_is_deterministic() -> None:
    first = derive_marketplace_configured_tool_execution_target_reference(_BINDING_A)
    second = derive_marketplace_configured_tool_execution_target_reference(_BINDING_A)
    assert first == second


def test_distinct_binding_episodes_produce_distinct_opaque_correlations() -> None:
    ref_a = derive_marketplace_configured_tool_execution_target_reference(_BINDING_A)
    ref_b = derive_marketplace_configured_tool_execution_target_reference(_BINDING_B)
    assert ref_a != ref_b


def test_configured_target_reference_does_not_embed_binding_operation_id() -> None:
    ref = derive_marketplace_configured_tool_execution_target_reference(_BINDING_A)
    assert _BINDING_A not in ref
    assert ref.startswith(_CONFIGURED_TARGET_PREFIX)
    suffix = ref[len(_CONFIGURED_TARGET_PREFIX) :]
    assert len(suffix) == 64
    assert suffix.isascii() and all(c in "0123456789abcdef" for c in suffix)


def test_configured_target_reference_does_not_encode_capability_identity() -> None:
    identity = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="tools.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="catalog.tools.sqlite.query",
    )
    ref = derive_marketplace_configured_tool_execution_target_reference(_BINDING_A)
    assert identity.logical_id not in ref
    assert identity.source_id not in ref
