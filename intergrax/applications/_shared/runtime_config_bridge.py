# © Artur Czarnecki. All rights reserved.

"""Bridge Tier-3 :class:`RuntimePolicyBundle` into Nexus :class:`RuntimeConfig` (Phase R-Policy)."""

from __future__ import annotations

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


def apply_policy_bundle_to_runtime_config(
    config: RuntimeConfig,
    bundle: RuntimePolicyBundle | None,
) -> RuntimeConfig:
    """Attach composed policy bundle; fill Nexus budget/plan-loop slots when unset."""
    if bundle is None:
        return config
    config.policy_bundle = bundle
    if bundle.budget is not None and config.budget_policy is None:
        config.budget_policy = bundle.budget
    if bundle.plan_loop is not None and config.plan_loop_policy is None:
        config.plan_loop_policy = bundle.plan_loop
    return config
