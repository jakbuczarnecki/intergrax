# © Artur Czarnecki. All rights reserved.

"""Load policy rule handlers from entry points (Phase DX-5.8)."""

from __future__ import annotations

from intergrax.core.plugins.discovery import (
    EP_POLICY_RULES,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry, PolicyRuleHandler


def load_policy_rule_plugins(registry: PolicyRuleRegistry) -> int:
    """Register handlers from ``intergrax.policy_rules`` entry points."""
    count = 0
    for spec in iter_entry_point_specs(EP_POLICY_RULES):
        loaded = load_entry_point_value(spec.value)
        instance = instantiate_entry_point_target(loaded)
        if not isinstance(instance, PolicyRuleHandler):
            raise TypeError(
                f"Policy rule entry point {spec.name!r} must return PolicyRuleHandler"
            )
        registry.register(instance)
        count += 1
    return count
