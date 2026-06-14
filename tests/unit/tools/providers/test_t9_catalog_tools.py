# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Sequence

import pytest

from intergrax.integrations.contracts.collaboration_suite import CalendarEvent, CollaborationSuite
from intergrax.integrations.contracts.workflow_orchestrator import (
    WorkflowOrchestratorBackend,
    WorkflowRunHandle,
    WorkflowRunStatus,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunError,
    RunMetadata,
    RunStats,
)
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationCreateEventInput,
    CollaborationReplyMessageInput,
)
from intergrax.tools.providers.collaboration.service import (
    collaboration_create_event,
    collaboration_reply_message,
)
from intergrax.tools.providers.harness.contracts import (
    HarnessCompareRunsInput,
    HarnessExportRunBundleInput,
)
from intergrax.tools.providers.harness.service import harness_compare_runs, harness_export_run_bundle
from intergrax.tools.providers.interaction.contracts import (
    InteractionGetLastInputInput,
    InteractionListSessionsInput,
)
from intergrax.tools.providers.interaction.service import interaction_get_last_input, interaction_list_sessions
from intergrax.tools.providers.notify.contracts import NotifyBatchMessageInput, NotifySendBatchInput
from intergrax.tools.providers.notify.service import notify_send_batch
from intergrax.tools.providers.websearch.invalidate_cache_contracts import WebsearchInvalidateCacheInput
from intergrax.tools.providers.websearch.invalidate_cache_service import perform_websearch_invalidate_cache
from intergrax.tools.providers.workflow.contracts import WorkflowCancelRunInput, WorkflowListRunsInput
from intergrax.tools.providers.workflow.service import workflow_cancel_run, workflow_list_runs
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeWorkflowOrchestrator:
    def list_runs(self, *, workflow_id: str = "", limit: int = 20) -> Sequence[WorkflowRunHandle]:
        del workflow_id, limit
        return [WorkflowRunHandle(run_id="wf-1", status="running", url="https://example/run/wf-1")]

    def cancel_run(self, run_id: str) -> WorkflowRunStatus:
        return WorkflowRunStatus(run_id=run_id, status="cancelled", conclusion="cancelled")


class FakeCollaborationSuite:
    def __init__(self) -> None:
        self.replied: list[tuple[str, str, str]] = []
        self.created: list[str] = []

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        self.replied.append((user_id, message_id, body))

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees: Sequence[str] = (),
    ) -> CalendarEvent:
        self.created.append(subject)
        return CalendarEvent(
            id="evt-1",
            subject=subject,
            start=start,
            end=end,
            location=location,
            organizer=user_id or "org",
        )


class FakeNotificationChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, message: object) -> None:
        self.messages.append(str(attribute_access.optional(message, "subject", "")))


class FakeWebSearchCache:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def invalidate_query_cache(self, *, query: str = "", clear_all: bool = False) -> int:
        self.calls.append((query, clear_all))
        return 2 if clear_all else 1


class FakeSessionStorage:
    def list_sessions(self, tenant_id: str, user_id: str, *, limit: int = 20) -> list[dict[str, str]]:
        del limit
        return [
            {
                "session_id": "sess-1",
                "tenant_id": tenant_id,
                "user_id": user_id,
                "updated_at_utc": "2026-06-07T12:00:00Z",
            }
        ]

    def get_last_user_input(self, tenant_id: str, session_id: str) -> str | None:
        del tenant_id
        if session_id == "sess-1":
            return "hello harness"
        return None

    def get_session_history(
        self,
        tenant_id: str,
        session_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, str]]:
        del tenant_id, limit
        return []


class InMemoryTraceReader:
    def __init__(self) -> None:
        self._baseline = PersistedRun(
            metadata=RunMetadata(
                run_id="run-a",
                session_id="sess-a",
                user_id="u-1",
                tenant_id="tenant-a",
                started_at_utc="2026-06-07T10:00:00Z",
                stats=RunStats(duration_ms=100, llm_usage={"input_tokens": 5}),
            ),
            events=[{"event_id": "e-1"}],
        )
        self._candidate = PersistedRun(
            metadata=RunMetadata(
                run_id="run-b",
                session_id="sess-b",
                user_id="u-1",
                tenant_id="tenant-a",
                started_at_utc="2026-06-07T10:05:00Z",
                stats=RunStats(duration_ms=150, llm_usage={"input_tokens": 8}),
                error=RunError(error_type=RuntimeErrorCode.INTERNAL_ERROR, message="boom"),
            ),
            events=[{"event_id": "e-1"}, {"event_id": "e-2"}],
        )

    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        if tenant_id != "tenant-a":
            raise KeyError("tenant")
        if run_id == "run-a":
            return self._baseline
        if run_id == "run-b":
            return self._candidate
        raise KeyError("run")


def test_workflow_list_runs() -> None:
    backend: WorkflowOrchestratorBackend = FakeWorkflowOrchestrator()  # type: ignore[assignment]
    ctx = ToolWiringContext(workflow_orchestrator=backend)
    out = workflow_list_runs(ctx, WorkflowListRunsInput(limit=5))
    assert out.total == 1
    assert out.runs[0].run_id == "wf-1"


def test_workflow_cancel_run() -> None:
    backend: WorkflowOrchestratorBackend = FakeWorkflowOrchestrator()  # type: ignore[assignment]
    ctx = ToolWiringContext(workflow_orchestrator=backend)
    out = workflow_cancel_run(ctx, WorkflowCancelRunInput(run_id="wf-9"))
    assert out.status == "cancelled"


def test_notify_send_batch() -> None:
    channel = FakeNotificationChannel()
    ctx = ToolWiringContext(notification_channel=channel)
    out = notify_send_batch(
        ctx,
        NotifySendBatchInput(
            messages=[
                NotifyBatchMessageInput(subject="one", body="a"),
                NotifyBatchMessageInput(subject="two", body="b"),
            ]
        ),
    )
    assert out.sent_count == 2
    assert out.failed_count == 0
    assert channel.messages == ["one", "two"]


def test_collaboration_reply_and_create_event() -> None:
    suite: CollaborationSuite = FakeCollaborationSuite()  # type: ignore[assignment]
    ctx = ToolWiringContext(collaboration_suite=suite)
    reply = collaboration_reply_message(
        ctx,
        CollaborationReplyMessageInput(user_id="u-1", message_id="msg-1", body="thanks"),
    )
    assert reply.replied is True
    event = collaboration_create_event(
        ctx,
        CollaborationCreateEventInput(
            user_id="u-1",
            subject="sync",
            start="2026-06-07T11:00:00Z",
            end="2026-06-07T12:00:00Z",
        ),
    )
    assert event.event is not None
    assert event.event.id == "evt-1"


def test_websearch_invalidate_cache() -> None:
    cache = FakeWebSearchCache()
    ctx = ToolWiringContext(websearch_executor=cache)
    out = perform_websearch_invalidate_cache(
        ctx,
        WebsearchInvalidateCacheInput(query="intergrax", clear_all=False),
    )
    assert out.used is True
    assert out.invalidated == 1
    assert cache.calls == [("intergrax", False)]


def test_harness_compare_and_export_run_bundle() -> None:
    ctx = ToolWiringContext(trace_reader=InMemoryTraceReader())
    compare = harness_compare_runs(
        ctx,
        HarnessCompareRunsInput(
            tenant_id="tenant-a",
            baseline_run_id="run-a",
            candidate_run_id="run-b",
        ),
    )
    assert compare.duration_delta_ms == 50
    assert compare.event_count_delta == 1
    assert compare.candidate.error_type == RuntimeErrorCode.INTERNAL_ERROR.value

    exported = harness_export_run_bundle(
        ctx,
        HarnessExportRunBundleInput(run_id="run-b", tenant_id="tenant-a", max_events=1),
    )
    assert exported.run_id == "run-b"
    assert exported.event_count == 1
    assert '"truncated": true' in exported.bundle_json


def test_interaction_session_tools() -> None:
    ctx = ToolWiringContext(session_storage=FakeSessionStorage())
    listed = interaction_list_sessions(
        ctx,
        InteractionListSessionsInput(tenant_id="tenant-a", user_id="u-1"),
    )
    assert listed.total == 1
    assert listed.sessions[0].session_id == "sess-1"

    last = interaction_get_last_input(
        ctx,
        InteractionGetLastInputInput(tenant_id="tenant-a", session_id="sess-1"),
    )
    assert last.found is True
    assert last.message == "hello harness"
