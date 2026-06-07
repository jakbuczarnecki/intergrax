# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.integrations.contracts.ci_cd import CheckSuiteRecord, CiCdBackend, WorkflowRunRecord
from intergrax.tools.providers.platform.contracts import PlatformCancelWorkflowRunInput, PlatformListWorkflowRunsInput
from intergrax.tools.providers.platform.service import platform_cancel_workflow_run, platform_list_workflow_runs
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeCiCdBackend:
    def get_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        return WorkflowRunRecord(id=run_id, status="completed")

    def list_check_suites(self, *, ref: str, limit: int = 20) -> Sequence[CheckSuiteRecord]:
        del ref, limit
        return []

    def list_workflow_runs(
        self,
        *,
        workflow_id: str = "",
        ref: str = "",
        limit: int = 20,
    ) -> Sequence[WorkflowRunRecord]:
        del workflow_id, ref, limit
        return [WorkflowRunRecord(id="run-1", name="ci", status="in_progress")]

    def cancel_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        return WorkflowRunRecord(id=run_id, status="cancelled", conclusion="cancelled")


def test_platform_list_workflow_runs() -> None:
    backend: CiCdBackend = FakeCiCdBackend()  # type: ignore[assignment]
    ctx = ToolWiringContext(ci_cd_backend=backend)
    out = platform_list_workflow_runs(ctx, PlatformListWorkflowRunsInput(limit=5))
    assert out.total == 1
    assert out.runs[0].id == "run-1"


def test_platform_cancel_workflow_run() -> None:
    backend: CiCdBackend = FakeCiCdBackend()  # type: ignore[assignment]
    ctx = ToolWiringContext(ci_cd_backend=backend)
    out = platform_cancel_workflow_run(ctx, PlatformCancelWorkflowRunInput(run_id="run-9"))
    assert out.status == "cancelled"
