# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-1C — dynamic ToolScopePolicy plan filtering."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _SelectiveScopePolicy:
    def __init__(self, *, allowed: set[str]) -> None:
        self._allowed = frozenset(allowed)

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return tool_id in self._allowed


class _AgentScopedDenyPolicy:
    def __init__(self, *, denied_agent: str, denied_tool: str) -> None:
        self._denied_agent = denied_agent
        self._denied_tool = denied_tool

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        if tool_id != self._denied_tool:
            return True
        return agent_id != self._denied_agent


def test_apply_scope_policy_filters_tool_ids() -> None:
    plan = ToolInvocationPlan.from_tool_ids(["tool.a", "tool.b", "tool.c"])
    policy = _SelectiveScopePolicy(allowed={"tool.a", "tool.c"})

    filtered = ToolAccessPolicy.apply_scope_policy(
        plan,
        scope_policy=policy,
        agent_id="customer-agent",
    )

    assert filtered.tool_ids == ("tool.a", "tool.c")


def test_apply_scope_policy_preserves_rag_alias_semantics() -> None:
    plan = ToolInvocationPlan.from_tool_ids(["rag", WEBSEARCH_QUERY_TOOL_ID])
    policy = _SelectiveScopePolicy(allowed={WEBSEARCH_QUERY_TOOL_ID})

    filtered = ToolAccessPolicy.apply_scope_policy(
        plan,
        scope_policy=policy,
        agent_id="agent-1",
    )

    assert filtered.tool_ids == (WEBSEARCH_QUERY_TOOL_ID,)


def test_apply_scope_policy_denies_rag_retrieve() -> None:
    plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    policy = _SelectiveScopePolicy(allowed=set())

    filtered = ToolAccessPolicy.apply_scope_policy(
        plan,
        scope_policy=policy,
        agent_id="agent-1",
    )

    assert filtered.tool_ids == ()


def test_apply_scope_policy_is_agent_aware() -> None:
    plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    policy = _AgentScopedDenyPolicy(
        denied_agent="customer-agent",
        denied_tool=RAG_RETRIEVE_TOOL_ID,
    )

    denied = ToolAccessPolicy.apply_scope_policy(
        plan,
        scope_policy=policy,
        agent_id="customer-agent",
    )
    allowed = ToolAccessPolicy.apply_scope_policy(
        plan,
        scope_policy=policy,
        agent_id="nexus",
    )

    assert denied.tool_ids == ()
    assert allowed.tool_ids == (RAG_RETRIEVE_TOOL_ID,)
