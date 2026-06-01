# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools_from_config
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

pytestmark = pytest.mark.gate


def test_explicit_allowed_tools_win_over_bundle() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"rag.retrieve"})
    config = RuntimeConfig(
        llm_adapter=MagicMock(),
        policy_bundle=RuntimePolicyBundle(tool_access=scope),
    )
    resolved = resolve_allowed_tools_from_config(
        config,
        explicit=["websearch.query"],
    )
    assert resolved == ["websearch.query"]


def test_bundle_static_scope_yields_sorted_allow_list() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"b.tool", "a.tool"})
    config = RuntimeConfig(
        llm_adapter=MagicMock(),
        policy_bundle=RuntimePolicyBundle(tool_access=scope),
    )
    resolved = resolve_allowed_tools_from_config(config)
    assert resolved == ["a.tool", "b.tool"]
