# © Artur Czarnecki. All rights reserved.

"""Tier-3 runtime policy bundle composition (Phase R-Policy.2)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


def build_runtime_policy_bundle(
    *,
    require_human_on_critical: bool = True,
    domain_fragments: dict[str, Any] | None = None,
) -> RuntimePolicyBundle:
    """Default harness policy bundle for lab and product hosts."""
    return RuntimePolicyBundle(
        require_human_on_critical=require_human_on_critical,
        domain_fragments=dict(domain_fragments or {}),
    )
