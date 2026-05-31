# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

import pytest

from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.jira.bundle import register_jira_tools
from intergrax.tools.providers.jira.contracts import JiraAddCommentInput, JiraGetIssueInput, JiraSearchTasksInput
from intergrax.tools.providers.jira.service import (
    build_jira_jql,
    jira_add_comment,
    jira_get_issue,
    jira_search_tasks,
)
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class FakeIssueTracker:
    def __init__(self) -> None:
        self.last_jql: Optional[str] = None

    def get_issue(self, issue_key: str) -> IssueRecord:
        return IssueRecord(
            key=issue_key,
            summary="Fix login bug",
            description="Steps to reproduce…",
            status="In Progress",
            assignee="alice",
            url=f"https://jira.example/browse/{issue_key}",
        )

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        return IssueComment(id="c-1", body=body, author="bot")

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        self.last_jql = jql
        return IssueSearchResult(
            issues=[
                IssueRecord(key="PROJ-1", summary="Task one", status="Open"),
                IssueRecord(key="PROJ-2", summary="Task two", status="Done"),
            ],
            total=2,
        )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_build_jira_jql() -> None:
    jql = build_jira_jql(
        JiraSearchTasksInput(project="PROJ", status="Open", assignee="alice", limit=10),
    )
    assert 'project = "PROJ"' in jql
    assert 'status = "Open"' in jql
    assert 'assignee = "alice"' in jql
    assert jql.endswith("order by updated DESC")


def test_jira_get_issue() -> None:
    ctx = ToolWiringContext(issue_tracker=FakeIssueTracker())
    out = jira_get_issue(ctx, JiraGetIssueInput(issue_key="PROJ-42"))
    assert out.key == "PROJ-42"
    assert out.summary == "Fix login bug"
    assert out.assignee == "alice"


def test_jira_add_comment() -> None:
    ctx = ToolWiringContext(issue_tracker=FakeIssueTracker())
    out = jira_add_comment(
        ctx,
        JiraAddCommentInput(issue_key="PROJ-42", body="Looks good."),
    )
    assert out.id == "c-1"
    assert out.body == "Looks good."
    assert out.issue_key == "PROJ-42"


def test_jira_search_tasks() -> None:
    tracker = FakeIssueTracker()
    ctx = ToolWiringContext(issue_tracker=tracker)
    out = jira_search_tasks(
        ctx,
        JiraSearchTasksInput(project="PROJ", status="Open", limit=5),
    )
    assert out.total == 2
    assert len(out.issues) == 2
    assert tracker.last_jql is not None
    assert 'project = "PROJ"' in out.jql


def test_jira_tracker_not_configured() -> None:
    with pytest.raises(RuntimeError, match="issue_tracker_not_configured"):
        jira_get_issue(ToolWiringContext(), JiraGetIssueInput(issue_key="X-1"))


def test_jira_tools_registered_in_catalog() -> None:
    register_default_tools()
    ids = list_catalog_tool_ids()
    assert "jira.get_issue" in ids
    assert "jira.add_comment" in ids
    assert "jira.search_tasks" in ids
    assert get_bundle("jira").tool_ids == (
        "jira.get_issue",
        "jira.add_comment",
        "jira.search_tasks",
    )


def test_jira_get_issue_via_runtime_invoker() -> None:
    ctx = ToolWiringContext(issue_tracker=FakeIssueTracker())
    registry = ToolRegistry()
    register_jira_tools(registry, ctx)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="jira_run")
    request = ToolExecutionRequest(
        run_id="jira_run",
        step_id="step/1",
        tool_id="jira.get_issue",
        input=JiraGetIssueInput(issue_key="PROJ-99"),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.key == "PROJ-99"


def test_build_registry_enables_jira_bundle() -> None:
    register_default_tools()
    ctx = ToolWiringContext(issue_tracker=FakeIssueTracker())
    registry = build_registry_from_profile(ToolProfile(enabled_bundles=["jira"]), ctx=ctx)
    assert registry.has("jira.get_issue")
    assert registry.has("jira.search_tasks")
