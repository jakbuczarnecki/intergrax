# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.http_client.allowlist.client import AllowlistHttpClient
from intergrax.memory.user_profile_memory import (
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.runtime.sandbox.sandbox_runtime import AGENT_BUILDER_SANDBOX_OPERATIONS
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.skills.core.contracts import SkillRiskTier
from intergrax.skills.resolver import ResolvedSkillPack
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.agent.contracts import AgentGetContractInput, AgentListAgentsInput
from intergrax.tools.providers.agent.service import agent_get_contract, agent_list_agents
from intergrax.tools.providers.catalog.contracts import CatalogDescribeToolInput, CatalogListToolsInput
from intergrax.tools.providers.catalog.service import catalog_describe_tool, catalog_list_tools
from intergrax.tools.providers.context_tool.contracts import ContextEstimateTokensInput, ContextSummarizeInput
from intergrax.tools.providers.context_tool.service import context_estimate_tokens, context_summarize
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.service import http_request
from intergrax.tools.providers.interaction.contracts import InteractionPostReplyInput
from intergrax.tools.providers.interaction.service import interaction_post_reply
from intergrax.tools.providers.issues.contracts import IssuesUpdateIssueInput
from intergrax.tools.providers.issues.service import issues_update_issue
from intergrax.tools.providers.ltm.contracts import LtmSearchInput, LtmWriteFactInput
from intergrax.tools.providers.ltm.service import ltm_search, ltm_write_fact
from intergrax.tools.providers.memory.contracts import MemorySearchInput
from intergrax.tools.providers.memory.service import memory_search
from intergrax.tools.providers.sandbox.contracts import CodeExecInput, SandboxListOperationsInput
from intergrax.tools.providers.sandbox.extended_service import code_exec, sandbox_list_operations
from intergrax.tools.providers.skill_tool.contracts import SkillResolveInput
from intergrax.tools.providers.skill_tool.service import skill_resolve
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
pytestmark = pytest.mark.unit


class _EchoHandler:
    def execute(self, request):
        return request.input


class _FakeAgentRegistry:
    def list_agent_ids(self) -> list[str]:
        return ["demo"]

    def get_agent_contract(self, agent_id: str) -> AgentContract:
        return AgentContract(
            id=agent_id,
            name="Demo Agent",
            description="Demo",
            capabilities=["demo.cap"],
        )


class _FakeUserProfileManager:
    def is_longterm_rag_enabled(self) -> bool:
        return False

    async def get_profile(self, user_id: str) -> UserProfile:
        _ = user_id
        return UserProfile(
            identity=UserIdentity(user_id=user_id),
            preferences=UserPreferences(),
            memory_entries=[
                UserProfileMemoryEntry(content="Intergrax harness builder", kind=MemoryKind.USER_FACT),
            ],
        )

    async def search_longterm_memory(self, user_id: str, query: str, *, top_k: int | None = None, score_threshold: float | None = None):
        _ = (user_id, query, top_k, score_threshold)
        return {"used_longterm": False, "hits": [], "scores": [], "debug": {"used": False, "reason": "disabled"}}

    async def add_memory_entry(self, user_id: str, entry_or_content, metadata=None):
        _ = (user_id, metadata)
        if isinstance(entry_or_content, UserProfileMemoryEntry):
            return entry_or_content
        return UserProfileMemoryEntry(content=str(entry_or_content))


class _FakeMemoryView:
    async def read(self, namespace: str, key: str):
        if namespace == "ns" and key == "alpha":
            return {"note": "hello intergrax"}
        return None

    async def write(self, namespace: str, key: str, value, *, policy=None):
        _ = (namespace, key, value, policy)

    async def list(self, namespace: str, prefix: str = ""):
        _ = prefix
        from intergrax.runtime.task_memory.models import TaskMemoryRecord

        return [
            TaskMemoryRecord(
                tenant_id="t1",
                task_id="task1",
                namespace=namespace,
                key="alpha",
                record_id="r1",
                updated_at_utc="now",
            )
        ]

    async def delete(self, namespace: str, key: str) -> bool:
        _ = (namespace, key)
        return True


class _UpdatingTracker:
    def get_issue(self, issue_key: str) -> IssueRecord:
        return IssueRecord(key=issue_key, summary="updated", status="done")

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        return IssueComment(id="1", body=body)

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return IssueSearchResult()

    def update_issue(self, issue_key: str, *, status=None, assignee=None, summary=None) -> IssueRecord:
        return IssueRecord(key=issue_key, summary=summary or "updated", status=status or "done")


class _NotifyChannel:
    def __init__(self) -> None:
        self.messages: list = []

    async def notify(self, message) -> None:
        self.messages.append(message)


def test_catalog_list_and_describe_tools() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="demo.tool",
            name="demo.tool",
            description="Demo tool",
            input_schema=CatalogListToolsInput,
            output_schema=CatalogListToolsInput,
            error_mapping={},
            side_effects=False,
            category="demo",
            risk_level=ToolRiskLevel.LOW,
            tags=("demo",),
        ),
        _EchoHandler(),
    )
    ctx = ToolWiringContext()
    listed = catalog_list_tools(ctx, CatalogListToolsInput(), registry=registry)
    assert listed.total == 1
    described = catalog_describe_tool(ctx, CatalogDescribeToolInput(tool_id="demo.tool"), registry=registry)
    assert described.found is True
    assert described.tool_id == "demo.tool"


class _FakeSkillResolver:
    def resolve(self, skill_ids: list[str]) -> ResolvedSkillPack:
        return ResolvedSkillPack(
            skill_ids=tuple(skill_ids),
            tool_ids=frozenset(["rag.retrieve"]),
            prompt_instruction_ids=frozenset(),
            policy_fragment_ids=frozenset(),
            risk_tier=SkillRiskTier.LOW,
        )


def test_agent_and_skill_introspection() -> None:
    ctx = ToolWiringContext(
        agent_registry=_FakeAgentRegistry(),
        skill_resolver=_FakeSkillResolver(),
    )
    agents = agent_list_agents(ctx, AgentListAgentsInput())
    assert agents.total == 1
    contract = agent_get_contract(ctx, AgentGetContractInput(agent_id="demo"))
    assert contract.found is True
    resolved = skill_resolve(ctx, SkillResolveInput(skill_ids=["harness.integration_bridge_smoke"]))
    assert "rag.retrieve" in resolved.tool_ids


def test_sandbox_code_exec_and_list_operations(tmp_path: Path) -> None:
    session = SandboxSession.create(
        tmp_path,
        tenant_id="t",
        task_id="task",
        allowed_operations=AGENT_BUILDER_SANDBOX_OPERATIONS,
    )
    ctx = ToolWiringContext(sandbox_session=session)
    ops = sandbox_list_operations(ctx, SandboxListOperationsInput())
    assert "run_python" in ops.operations
    result = code_exec(ctx, CodeExecInput(code="print(42)"))
    assert result.success is True
    assert "42" in result.output.get("stdout", "")


def test_ltm_memory_context_and_http_tools() -> None:
    ctx = ToolWiringContext(
        user_profile_manager=_FakeUserProfileManager(),
        memory_view=_FakeMemoryView(),
        http_client=AllowlistHttpClient(allowed_hosts=frozenset({"example.com"})),
    )
    ltm_hits = ltm_search(ctx, LtmSearchInput(user_id="u1", query="intergrax"))
    assert ltm_hits.used is True
    written = ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="likes python"))
    assert written.written is True
    mem_hits = memory_search(ctx, MemorySearchInput(namespace="ns", query="intergrax"))
    assert mem_hits.total >= 1
    summary = context_summarize(ctx, ContextSummarizeInput(text="word " * 2000, max_tokens=64))
    assert summary.trimmed is True
    tokens = context_estimate_tokens(ctx, ContextEstimateTokensInput(text="abcd"))
    assert tokens.token_estimate >= 1
    denied = http_request(ctx, HttpRequestInput(method="GET", url="https://evil.com/"))
    assert denied.success is False


def test_interaction_reply_and_issue_update() -> None:
    channel = _NotifyChannel()
    ctx = ToolWiringContext(notification_channel=channel, issue_tracker=_UpdatingTracker())
    reply = interaction_post_reply(
        ctx,
        InteractionPostReplyInput(tenant_id="t1", channel="log", body="hello", session_id="s1"),
    )
    assert reply.sent is True
    updated = issues_update_issue(ctx, IssuesUpdateIssueInput(issue_key="X-1", summary="New title"))
    assert updated.issue.summary == "New title"
