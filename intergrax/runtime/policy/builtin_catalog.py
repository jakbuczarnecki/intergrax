# © Artur Czarnecki. All rights reserved.

"""Canonical built-in Policy Catalog definitions and composition (Governed Execution G2C-2B)."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
    ToolInvocationControlConfig,
)
from intergrax.runtime.policy.catalog import PolicyCatalog
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule

TOOL_INVOCATION_CONTROL_POLICY_ID = "tool_invocation_control"
TOOL_INVOCATION_CONTROL_VERSION = "1"
TOOL_INVOCATION_CONTROL_HANDLER_ID = "deny_tool"
TOOL_INVOCATION_CONTROL_RESOURCE_KIND = "tool"


class ToolInvocationControlCompositionError(Exception):
    """Raised when catalog definition or rule composition inputs are invalid."""


def _tool_invocation_control_definition() -> PolicyDefinition:
    return PolicyDefinition(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
        display_name="Tool Invocation Control",
        description=(
            "Controls tool invocation outcome: allow execution, deny before side effects, "
            "or require human approval."
        ),
        handler_id=TOOL_INVOCATION_CONTROL_HANDLER_ID,
        configuration_contract_id=TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
        source=PolicyDefinitionSource.BUILT_IN,
    )


def built_in_policy_definitions() -> tuple[PolicyDefinition, ...]:
    """Return canonical built-in policy definitions in deterministic order."""
    return (_tool_invocation_control_definition(),)


def build_builtin_policy_catalog() -> PolicyCatalog:
    """Build the canonical immutable built-in Policy Catalog."""
    return build_policy_catalog()


def build_policy_catalog(
    *,
    plugin_definitions: Iterable[PolicyDefinition] = (),
) -> PolicyCatalog:
    """Compose built-in and validated plugin PolicyDefinition values."""
    return PolicyCatalog((*built_in_policy_definitions(), *plugin_definitions))


def _validate_builtin_tool_invocation_control_definition(definition: PolicyDefinition) -> None:
    if definition.policy_id != TOOL_INVOCATION_CONTROL_POLICY_ID:
        raise ToolInvocationControlCompositionError(
            f"expected policy_id {TOOL_INVOCATION_CONTROL_POLICY_ID!r}, "
            f"got {definition.policy_id!r}"
        )
    if definition.version != TOOL_INVOCATION_CONTROL_VERSION:
        raise ToolInvocationControlCompositionError(
            f"expected version {TOOL_INVOCATION_CONTROL_VERSION!r}, "
            f"got {definition.version!r}"
        )
    if definition.configuration_contract_id != TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID:
        raise ToolInvocationControlCompositionError(
            "expected configuration_contract_id "
            f"{TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID!r}, "
            f"got {definition.configuration_contract_id!r}"
        )
    if definition.handler_id != TOOL_INVOCATION_CONTROL_HANDLER_ID:
        raise ToolInvocationControlCompositionError(
            f"expected handler_id {TOOL_INVOCATION_CONTROL_HANDLER_ID!r}, "
            f"got {definition.handler_id!r}"
        )
    if definition.source is not PolicyDefinitionSource.BUILT_IN:
        raise ToolInvocationControlCompositionError(
            f"expected source {PolicyDefinitionSource.BUILT_IN!r}, got {definition.source!r}"
        )


def _normalize_rule_id(rule_id: str) -> str:
    normalized = rule_id.strip()
    if not normalized:
        raise ToolInvocationControlCompositionError("rule_id must be non-empty")
    return normalized


def compose_tool_invocation_control_rule(
    *,
    rule_id: str,
    definition: PolicyDefinition,
    config: ToolInvocationControlConfig,
) -> DeclarativePolicyRule:
    """Compose a configured declarative rule from catalog definition and typed config."""
    _validate_builtin_tool_invocation_control_definition(definition)
    normalized_rule_id = _normalize_rule_id(rule_id)
    return DeclarativePolicyRule(
        rule_id=normalized_rule_id,
        handler_id=definition.handler_id,
        resource_kind=TOOL_INVOCATION_CONTROL_RESOURCE_KIND,
        resource_id=config.tool_id,
        action=config.action,
        conditions={},
    )
