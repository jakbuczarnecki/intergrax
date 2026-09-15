# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.policy.tool_policy_resolution import (
    ToolPolicyResolutionError,
    resolve_allowed_tools,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

pytestmark = pytest.mark.gate


def test_no_upstream_and_no_explicit_returns_none() -> None:
    assert resolve_allowed_tools() is None
    assert resolve_allowed_tools(explicit=None) is None
    assert resolve_allowed_tools(upstream_policy=None, explicit=None) is None


def test_upstream_only_yields_sorted_allow_list() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"b.tool", "a.tool"})
    resolved = resolve_allowed_tools(upstream_policy=scope)
    assert resolved == ["a.tool", "b.tool"]


def test_upstream_empty_allow_list_is_not_unconstrained() -> None:
    scope = StaticToolScopePolicy(allowed_tools=set())
    resolved = resolve_allowed_tools(upstream_policy=scope)
    assert resolved == []
    assert resolved is not None


def test_explicit_only_preserves_caller_scope() -> None:
    explicit = ("b.tool", "a.tool")
    resolved = resolve_allowed_tools(explicit=explicit)
    assert resolved == explicit


def test_explicit_intersects_upstream_allow_list() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"a.tool", "b.tool"})
    resolved = resolve_allowed_tools(
        upstream_policy=scope,
        explicit=["b.tool", "c.tool"],
    )
    assert resolved == ["b.tool"]


def test_empty_intersection_returns_empty_list_not_none() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"a.tool"})
    resolved = resolve_allowed_tools(upstream_policy=scope, explicit=["b.tool"])
    assert resolved == []
    assert resolved is not None


def test_explicit_empty_returns_empty_without_upstream() -> None:
    assert resolve_allowed_tools(explicit=[]) == []


def test_explicit_empty_wins_over_nonempty_upstream() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"a.tool", "b.tool"})
    assert resolve_allowed_tools(upstream_policy=scope, explicit=[]) == []


def test_intersection_order_is_deterministic() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"z.tool", "a.tool", "m.tool"})
    resolved = resolve_allowed_tools(
        upstream_policy=scope,
        explicit=["m.tool", "a.tool", "x.tool"],
    )
    assert resolved == ["a.tool", "m.tool"]


def test_explicit_cannot_expand_upstream_policy_regression() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"tool.read"})
    resolved = resolve_allowed_tools(
        upstream_policy=scope,
        explicit=["tool.read", "tool.delete"],
    )
    assert resolved == ["tool.read"]
    assert "tool.delete" not in resolved


def test_explicit_allowed_tools_do_not_win_over_bundle() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"rag.retrieve"})
    resolved = resolve_allowed_tools(
        upstream_policy=scope,
        explicit=["websearch.query"],
    )
    assert resolved == []


class _CustomEnumeratedPolicy:
    def __init__(self, allowed: set[str]) -> None:
        self._allowed = frozenset(allowed)

    def __bool__(self) -> bool:
        return False

    def allowed_tool_ids(self) -> frozenset[str]:
        return self._allowed

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return tool_id in self._allowed


def test_custom_enumerated_policy_is_used_without_type_check() -> None:
    policy = _CustomEnumeratedPolicy({"x.tool"})
    resolved = resolve_allowed_tools(upstream_policy=policy, explicit=["x.tool", "y.tool"])
    assert resolved == ["x.tool"]


def test_falsey_custom_policy_is_not_ignored() -> None:
    policy = _CustomEnumeratedPolicy({"keep.me"})
    assert not policy
    resolved = resolve_allowed_tools(upstream_policy=policy)
    assert resolved == ["keep.me"]


class _NonEnumerablePolicy:
    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return True


def test_non_enumerable_upstream_policy_fails_closed() -> None:
    with pytest.raises(ToolPolicyResolutionError):
        resolve_allowed_tools(upstream_policy=_NonEnumerablePolicy())


def test_neutral_bundle_constructs_without_nexus_imports() -> None:
    bundle = RuntimePolicyBundle()
    assert bundle.tool_access is None
