# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools_from_config
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

pytestmark = pytest.mark.gate


def _config_with_scope(*allowed: str) -> RuntimeConfig:
    scope = StaticToolScopePolicy(allowed_tools=set(allowed))
    return RuntimeConfig(
        llm_adapter=MagicMock(),
        policy_bundle=RuntimePolicyBundle(tool_access=scope),
    )


def test_no_upstream_and_no_explicit_returns_none() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())
    assert resolve_allowed_tools_from_config(config) is None
    assert resolve_allowed_tools_from_config(config, explicit=None) is None


def test_upstream_only_yields_sorted_allow_list() -> None:
    resolved = resolve_allowed_tools_from_config(_config_with_scope("b.tool", "a.tool"))
    assert resolved == ["a.tool", "b.tool"]


def test_explicit_only_preserves_caller_scope() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())
    explicit = ("b.tool", "a.tool")
    resolved = resolve_allowed_tools_from_config(config, explicit=explicit)
    assert resolved == explicit


def test_explicit_intersects_upstream_allow_list() -> None:
    config = _config_with_scope("a.tool", "b.tool")
    resolved = resolve_allowed_tools_from_config(
        config,
        explicit=["b.tool", "c.tool"],
    )
    assert resolved == ["b.tool"]


def test_empty_intersection_returns_empty_list_not_none() -> None:
    config = _config_with_scope("a.tool")
    resolved = resolve_allowed_tools_from_config(config, explicit=["b.tool"])
    assert resolved == []
    assert resolved is not None


def test_intersection_order_is_deterministic() -> None:
    config = _config_with_scope("z.tool", "a.tool", "m.tool")
    resolved = resolve_allowed_tools_from_config(
        config,
        explicit=["m.tool", "a.tool", "x.tool"],
    )
    assert resolved == ["a.tool", "m.tool"]


def test_explicit_cannot_expand_upstream_policy_regression() -> None:
    """Regression: explicit caller scope must not replace a stricter bundle allow-list."""
    config = _config_with_scope("tool.read")
    resolved = resolve_allowed_tools_from_config(
        config,
        explicit=["tool.read", "tool.delete"],
    )
    assert resolved == ["tool.read"]
    assert "tool.delete" not in resolved


def test_explicit_allowed_tools_do_not_win_over_bundle() -> None:
    """Guards against the pre-P0-SAFETY-1 bug where explicit replaced upstream."""
    config = _config_with_scope("rag.retrieve")
    resolved = resolve_allowed_tools_from_config(
        config,
        explicit=["websearch.query"],
    )
    assert resolved == []
