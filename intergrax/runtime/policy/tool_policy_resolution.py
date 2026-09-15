# © Artur Czarnecki. All rights reserved.

"""Resolve effective tool allow-lists from neutral policy inputs (Phase R-Policy, GR-4-R1)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.runtime.tools.scope_policy import (
    ToolAllowListEnumeration,
    ToolScopePolicy,
)


class ToolPolicyResolutionError(ValueError):
    """Raised when upstream policy cannot be interpreted (fail-closed)."""


def _upstream_allow_list(upstream_policy: ToolScopePolicy) -> list[str]:
    if not isinstance(upstream_policy, ToolAllowListEnumeration):
        raise ToolPolicyResolutionError(
            "upstream_tool_policy_not_enumerable: "
            "policy must implement allowed_tool_ids() for list intersection",
        )
    try:
        raw = upstream_policy.allowed_tool_ids()
    except Exception as exc:  # noqa: BLE001 — fail-closed boundary
        raise ToolPolicyResolutionError(
            "upstream_tool_policy_allowed_tool_ids_failed",
        ) from exc
    if not isinstance(raw, frozenset):
        raise ToolPolicyResolutionError(
            "upstream_tool_policy_allowed_tool_ids_invalid_type",
        )
    return sorted(raw)


def resolve_allowed_tools(
    *,
    upstream_policy: ToolScopePolicy | None = None,
    explicit: Sequence[str] | None = None,
) -> Sequence[str] | None:
    """
    Resolve effective tool allow-list from neutral policy inputs.

    Semantics:
    - ``upstream_policy is None``: no upstream restriction (not the same as empty allow-list).
    - ``upstream_policy`` with empty enumeration: zero tools allowed when explicit is absent.
    - ``explicit is None``: no caller narrowing; use upstream only when present.
    - ``explicit`` empty sequence: zero tools allowed (fail-closed), regardless of upstream.
    - When both apply, result is sorted intersection; empty intersection is ``[]``, not ``None``.
    """
    if explicit is not None and len(explicit) == 0:
        return []

    upstream: list[str] | None = None
    if upstream_policy is not None:
        upstream = _upstream_allow_list(upstream_policy)

    if upstream is None and explicit is None:
        return None
    if upstream is None:
        return explicit
    if explicit is None:
        return upstream
    return sorted(set(upstream) & set(explicit))
