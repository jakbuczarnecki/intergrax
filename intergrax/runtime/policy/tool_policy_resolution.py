# © Artur Czarnecki. All rights reserved.

"""Resolve effective tool allow-lists from RuntimeConfig policy bundle (Phase R-Policy)."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy


def resolve_allowed_tools_from_config(
    config: RuntimeConfig,
    *,
    explicit: Optional[Sequence[str]] = None,
) -> Optional[Sequence[str]]:
    """
    Resolve effective tool allow-list from Tier-3 ``RuntimePolicyBundle.tool_access``.

    Caller ``explicit`` scope narrows (intersects with) upstream policy; it never
    expands a stricter bundle allow-list. A ``StaticToolScopePolicy`` on the bundle
    yields a sorted upstream allow-list for :class:`ToolAccessPolicy`. When both
    authorities apply, the result is their sorted intersection; an empty intersection
    is ``[]`` (zero allowed tools), not ``None``.
    """
    upstream: Optional[Sequence[str]] = None
    bundle = config.policy_bundle
    if bundle is not None and bundle.tool_access is not None:
        access = bundle.tool_access
        if isinstance(access, StaticToolScopePolicy):
            upstream = sorted(access.allowed_tool_ids())

    if upstream is None and explicit is None:
        return None
    if upstream is None:
        return explicit
    if explicit is None:
        return upstream
    return sorted(set(upstream) & set(explicit))
