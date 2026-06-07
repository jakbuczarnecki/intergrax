# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.integrations.contracts.object_storage import StoredObject
from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationMode, OnlineEvaluationObservation
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.providers.eval.contracts import EvalExportObservationsInput
from intergrax.tools.providers.eval.service import eval_export_observations
from intergrax.tools.providers.interaction.contracts import InteractionGetSessionHistoryInput
from intergrax.tools.providers.interaction.service import interaction_get_session_history
from intergrax.tools.providers.memory.contracts import MemoryDeleteKeyInput
from intergrax.tools.providers.memory.service import memory_delete_key
from intergrax.tools.providers.message_bus.contracts import MessageBusPurgeCompletedInput
from intergrax.tools.providers.message_bus.service import message_bus_purge_completed
from intergrax.tools.providers.notify.contracts import NotifyScheduleInput
from intergrax.tools.providers.notify.service import notify_schedule
from intergrax.tools.providers.pagerduty.contracts import PagerDutyAcknowledgeIncidentInput
from intergrax.tools.providers.pagerduty.service import pagerduty_acknowledge_incident
from intergrax.tools.providers.records.contracts import RecordsCountInput
from intergrax.tools.providers.records.service import records_count
from intergrax.tools.providers.storage.contracts import StorageExistsInput
from intergrax.tools.providers.storage.service import storage_exists
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceExportArtifactInput,
    WorkspaceImportArtifactInput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.service import (
    workspace_export_artifact,
    workspace_import_artifact,
    workspace_write_file,
)
from intergrax.tools.registry.scheduled_notification_binding import InMemoryScheduledNotificationStore
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def shadow_workspace(tmp_path: Path) -> ShadowWorkspace:
    return ShadowWorkspace.create(tmp_path, tenant_id="tenant-1", task_id="task-1")


class FakeObjectStorage:
    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata=None,
    ) -> None:
        del metadata
        self._objects[key] = StoredObject(key=key, body=body, content_type=content_type, size_bytes=len(body))

    def get(self, key: str) -> StoredObject | None:
        return self._objects.get(key)

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def presigned_url(self, key: str, *, expires_in_seconds: int = 3600, method: str = "GET") -> str:
        del expires_in_seconds, method
        return f"https://example/{key}"

    def close(self) -> None:
        return None


class FakeDocumentStore:
    def query(self, partition_key: str, *, limit: int = 100, row_key_prefix: str | None = None):
        del limit, row_key_prefix
        from intergrax.integrations.contracts.document_store import DocumentQueryResult

        return DocumentQueryResult(total=7 if partition_key == "pk-1" else 0)


class FakeMessageBus:
    def purge_completed(self, tenant_id: str, *, older_than_seconds: int = 0) -> int:
        del older_than_seconds
        return 3 if tenant_id == "tenant-a" else 0


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
        if session_id != "sess-1":
            return []
        return [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]


class FakeMemoryView:
    async def delete(self, namespace: str, key: str) -> bool:
        del namespace
        return key == "k1"


class FakeEvalRegistry:
    def append(self, observation: object) -> None:
        del observation

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        return [
            OnlineEvaluationObservation(
                observation_id="obs-1",
                run_id="run-1",
                agent_id="agent-1",
                mode=OnlineEvaluationMode.SHADOW,
                scenario_id="release-a",
                passed=True,
                score=0.9,
            )
        ]


class FakePagerDutyChannel:
    def __init__(self) -> None:
        self.acked: list[str] = []

    def trigger_incident(
        self,
        *,
        summary: str,
        severity: str = "error",
        source: str = "intergrax",
        custom_details=None,
        dedup_key: str | None = None,
    ) -> str:
        del summary, severity, source, custom_details
        return dedup_key or "dedup-generated"

    def acknowledge_incident(self, *, dedup_key: str, note: str | None = None) -> None:
        del note
        self.acked.append(dedup_key)


def test_workspace_export_and_import(shadow_workspace: ShadowWorkspace) -> None:
    storage = FakeObjectStorage()
    ctx = ToolWiringContext(shadow_workspace=shadow_workspace, object_storage=storage)
    workspace_write_file(ctx, WorkspaceWriteFileInput(path="out/report.txt", content="payload"))
    exported = workspace_export_artifact(
        ctx,
        WorkspaceExportArtifactInput(path="out/report.txt", storage_key="exports/report.txt"),
    )
    assert exported.exported is True
    assert exported.size_bytes == len("payload".encode())

    other = ShadowWorkspace.create(shadow_workspace.root.parent, tenant_id="tenant-2", task_id="task-2")
    import_ctx = ToolWiringContext(shadow_workspace=other, object_storage=storage)
    imported = workspace_import_artifact(
        import_ctx,
        WorkspaceImportArtifactInput(storage_key="exports/report.txt", path="in/report.txt"),
    )
    assert imported.imported is True
    assert imported.size_bytes == len("payload".encode())


def test_notify_schedule() -> None:
    ctx = ToolWiringContext(scheduled_notification_store=InMemoryScheduledNotificationStore())
    out = notify_schedule(
        ctx,
        NotifyScheduleInput(
            subject="maintenance",
            body="window starts",
            deliver_at_utc="2026-06-08T00:00:00Z",
        ),
    )
    assert out.scheduled is True
    assert out.schedule_id.startswith("sched_")


def test_interaction_get_session_history() -> None:
    ctx = ToolWiringContext(session_storage=FakeSessionStorage())
    out = interaction_get_session_history(
        ctx,
        InteractionGetSessionHistoryInput(tenant_id="tenant-a", session_id="sess-1"),
    )
    assert out.total == 2
    assert out.messages[0].role == "user"


def test_eval_export_observations() -> None:
    ctx = ToolWiringContext(evaluation_registry=FakeEvalRegistry())
    out = eval_export_observations(ctx, EvalExportObservationsInput(limit=10))
    payload = json.loads(out.export_json)
    assert out.observation_count == 1
    assert payload["total"] == 1


def test_storage_exists() -> None:
    storage = FakeObjectStorage()
    storage.put("obj-1", b"abc", content_type="text/plain")
    ctx = ToolWiringContext(object_storage=storage)
    missing = storage_exists(ctx, StorageExistsInput(key="missing"))
    assert missing.exists is False
    found = storage_exists(ctx, StorageExistsInput(key="obj-1"))
    assert found.exists is True
    assert found.size_bytes == 3


def test_memory_delete_key() -> None:
    ctx = ToolWiringContext(memory_view=FakeMemoryView())
    out = memory_delete_key(ctx, MemoryDeleteKeyInput(namespace="ns", key="k1"))
    assert out.deleted is True


def test_pagerduty_acknowledge_incident() -> None:
    channel = FakePagerDutyChannel()
    ctx = ToolWiringContext(notification_channel=channel)
    out = pagerduty_acknowledge_incident(
        ctx,
        PagerDutyAcknowledgeIncidentInput(dedup_key="dedup-1"),
    )
    assert out.acknowledged is True
    assert channel.acked == ["dedup-1"]


def test_message_bus_purge_completed() -> None:
    ctx = ToolWiringContext(message_bus=FakeMessageBus())
    out = message_bus_purge_completed(ctx, MessageBusPurgeCompletedInput(tenant_id="tenant-a"))
    assert out.purged_count == 3


def test_records_count() -> None:
    ctx = ToolWiringContext(document_store=FakeDocumentStore())
    out = records_count(ctx, RecordsCountInput(partition_key="pk-1"))
    assert out.total == 7
