# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from intergrax.integrations.contracts.workflow_orchestrator import WorkflowRunHandle, WorkflowRunStatus
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.ingest_contracts import RagScheduleIngestJobInput
from intergrax.tools.providers.rag.ingest_job_service import perform_rag_schedule_ingest_job
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.triggered: list[dict[str, str]] = []
        self._runs: list[WorkflowRunHandle] = []

    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str] | None = None) -> WorkflowRunHandle:
        params = dict(parameters or {})
        handle = WorkflowRunHandle(
            run_id=f"run-{len(self._runs) + 1}",
            status="pending",
            url=f"https://wf/{workflow_id}",
            metadata={"idempotency_key": params.get("idempotency_key", "")},
        )
        self.triggered.append(params)
        self._runs.append(handle)
        return handle

    def poll_status(self, run_id: str) -> WorkflowRunStatus:
        return WorkflowRunStatus(run_id=run_id, status="pending")

    def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
        return f"log:{run_id}"

    def list_runs(
        self,
        *,
        workflow_id: str = "",
        limit: int = 20,
    ) -> Sequence[WorkflowRunHandle]:
        del limit
        if workflow_id:
            return [run for run in self._runs if run.run_id.startswith("run-")]
        return list(self._runs)

    def cancel_run(self, run_id: str) -> WorkflowRunStatus:
        return WorkflowRunStatus(run_id=run_id, status="cancelled")


def test_schedule_ingest_job_triggers_workflow(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_text("content", encoding="utf-8")
    orchestrator = _FakeOrchestrator()

    out = perform_rag_schedule_ingest_job(
        ToolWiringContext(
            workflow_orchestrator=orchestrator,
            rag_profile=RagProfile(async_ingest_workflow_id="rag-ingest-batch"),
        ),
        RagScheduleIngestJobInput(source_path=str(source), tenant_id="tenant-a"),
    )

    assert out.used is True
    assert out.reason == "scheduled"
    assert out.run_id == "run-1"
    assert out.idempotency_key.startswith("rag-ingest-")
    assert orchestrator.triggered[0]["job_type"] == "rag.ingest"
    assert orchestrator.triggered[0]["tenant_id"] == "tenant-a"


def test_schedule_ingest_job_is_idempotent_for_active_run(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_text("content", encoding="utf-8")
    orchestrator = _FakeOrchestrator()
    ctx = ToolWiringContext(
        workflow_orchestrator=orchestrator,
        rag_profile=RagProfile(async_ingest_workflow_id="rag-ingest-batch"),
    )
    params = RagScheduleIngestJobInput(source_path=str(source), tenant_id="tenant-a")

    first = perform_rag_schedule_ingest_job(ctx, params)
    second = perform_rag_schedule_ingest_job(ctx, params)

    assert first.used is True
    assert second.used is True
    assert second.reason == "idempotent_reuse"
    assert second.run_id == first.run_id
    assert len(orchestrator.triggered) == 1


def test_schedule_ingest_job_requires_orchestrator(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_text("content", encoding="utf-8")

    out = perform_rag_schedule_ingest_job(
        ToolWiringContext(),
        RagScheduleIngestJobInput(source_path=str(source)),
    )

    assert out.used is False
    assert out.reason == "workflow_orchestrator_not_configured"
