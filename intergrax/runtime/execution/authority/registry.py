# © Artur Czarnecki. All rights reserved.

"""Entry-point registry and resolver for ``ExecutionAuthorityPolicy`` plugins (UE-8P2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.core.plugins.discovery import (
    EP_EXECUTION_AUTHORITY_POLICIES,
    get_entry_point_spec,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)
from intergrax.runtime.execution.authority.policy import (
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig

_ENTRY_POINT_GROUP = EP_EXECUTION_AUTHORITY_POLICIES


class ExecutionAuthorityPolicyConfigurationError(RuntimeError):
    """Raised when an explicitly configured authority policy cannot be loaded."""


def load_execution_authority_policy(policy_id: str) -> ExecutionAuthorityPolicy | None:
    """Load a policy by entry-point name from ``intergrax.execution_authority_policies``."""
    spec = get_entry_point_spec(_ENTRY_POINT_GROUP, policy_id)
    if spec is None:
        return None
    loaded = load_entry_point_value(spec.value)
    instance = instantiate_entry_point_target(loaded)
    if not isinstance(instance, ExecutionAuthorityPolicy):
        raise TypeError(
            f"execution authority entry point {spec.name!r} must return "
            "ExecutionAuthorityPolicy"
        )
    return instance


def list_execution_authority_policy_ids() -> tuple[str, ...]:
    """Return registered entry-point policy ids (sorted)."""
    return tuple(spec.name for spec in iter_entry_point_specs(_ENTRY_POINT_GROUP))


def resolve_execution_authority_policy(
    *,
    policy_override: ExecutionAuthorityPolicy | None = None,
    entry_point_policy_id: str | None = None,
) -> ExecutionAuthorityPolicy:
    """
    Resolve authority policy from explicit instance, entry-point id, or platform default.

    Explicit ``entry_point_policy_id`` fails closed when the plugin is missing or invalid.
    """
    if policy_override is not None:
        if not isinstance(policy_override, ExecutionAuthorityPolicy):
            raise TypeError("policy_override must satisfy ExecutionAuthorityPolicy")
        return policy_override
    if entry_point_policy_id:
        loaded = load_execution_authority_policy(entry_point_policy_id)
        if loaded is None:
            raise ExecutionAuthorityPolicyConfigurationError(
                f"execution authority policy entry point "
                f"{entry_point_policy_id!r} not found"
            )
        return loaded
    return DefaultStrictAuthorityPolicy()


def resolve_execution_authority_policy_from_runtime_config(
    config: RuntimeConfig,
) -> ExecutionAuthorityPolicy:
    """Resolve authority policy from ``RuntimeConfig`` execution authority fields."""
    return resolve_execution_authority_policy(
        policy_override=config.execution_authority_policy,
        entry_point_policy_id=config.execution_authority_policy_id,
    )
