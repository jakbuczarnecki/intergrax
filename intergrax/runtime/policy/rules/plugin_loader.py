# © Artur Czarnecki. All rights reserved.

"""Load policy rule handlers from entry points (Phase DX-5.8)."""

from __future__ import annotations

from importlib.metadata import entry_points

from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry, PolicyRuleHandler


def load_policy_rule_plugins(registry: PolicyRuleRegistry) -> int:
    """Register handlers from ``intergrax.policy_rules`` entry points."""
    try:
        eps = entry_points(group="intergrax.policy_rules")
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group="intergrax.policy_rules")
    count = 0
    for ep in eps:
        handler = ep.load()
        if isinstance(handler, type):
            instance: PolicyRuleHandler = handler()
        else:
            instance = handler
        registry.register(instance)
        count += 1
    return count
