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
    Merge explicit caller allow-list with Tier-3 ``RuntimePolicyBundle.tool_access``.

    When ``explicit`` is provided it wins. Otherwise a ``StaticToolScopePolicy`` on the
    bundle yields a sorted allow-list for :class:`ToolAccessPolicy`.
    """
    if explicit is not None:
        return explicit
    bundle = config.policy_bundle
    if bundle is None or bundle.tool_access is None:
        return None
    access = bundle.tool_access
    if isinstance(access, StaticToolScopePolicy):
        return sorted(access.allowed_tool_ids())
    return None
