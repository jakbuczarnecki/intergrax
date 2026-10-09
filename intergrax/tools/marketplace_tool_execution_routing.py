# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace Tool execution routing ids and canonical execution target construction."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)

MARKETPLACE_TOOL_EXECUTION_HANDLER_ID: Final = "marketplace.tool.execution.v1"

MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID: Final = (
    "marketplace.tool.configured_capability_binding.v1"
)

_CONFIGURED_TARGET_PREFIX: Final = "marketplace-configured-tool:v1:"


def derive_marketplace_configured_tool_execution_target_reference(
    binding_operation_id: str,
) -> str:
    normalized = require_non_empty_text(binding_operation_id, label="binding_operation_id")
    return f"{_CONFIGURED_TARGET_PREFIX}{normalized}"


def derive_marketplace_configured_tool_execution_intent_target_correlation(
    binding_operation_id: str,
) -> str:
    """Opaque durable intent link to configured binding — not capability identity."""
    return derive_marketplace_configured_tool_execution_target_reference(binding_operation_id)


def parse_marketplace_configured_tool_execution_target_reference(
    reference: str,
) -> str | None:
    if not reference.startswith(_CONFIGURED_TARGET_PREFIX):
        return None
    binding_operation_id = reference[len(_CONFIGURED_TARGET_PREFIX) :]
    try:
        return require_non_empty_text(binding_operation_id, label="binding_operation_id")
    except (TypeError, ValueError):
        return None


def build_marketplace_tool_execution_target(
    *,
    execution_target_reference: str,
    binding_provider_id: str,
    qualified_subject_reference: str,
) -> QualifiedCapabilityExecutionTarget:
    return QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference,
        binding_provider_id=binding_provider_id,
        execution_handler_id=MARKETPLACE_TOOL_EXECUTION_HANDLER_ID,
        qualified_subject_reference=qualified_subject_reference,
    )


__all__ = [
    "MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID",
    "MARKETPLACE_TOOL_EXECUTION_HANDLER_ID",
    "build_marketplace_tool_execution_target",
    "derive_marketplace_configured_tool_execution_intent_target_correlation",
    "derive_marketplace_configured_tool_execution_target_reference",
    "parse_marketplace_configured_tool_execution_target_reference",
]
