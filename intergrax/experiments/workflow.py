# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Experiment workflow helpers for notebooks and scripts (Phase D.4, §35).

Wraps experiment registry + NexusLoop + trace persistence into a single session API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
)
from intergrax.experiments.store import SQLiteExperimentStore, open_experiment_store
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState


def ensure_repo_root_on_path(start: Path | None = None) -> Path:
    """
    Add repository root to ``sys.path`` when running from a notebook.

    Looks for a directory containing both ``intergrax/`` and ``agents/``.
    """
    probe = (start or Path.cwd()).resolve()
    for candidate in (probe, *probe.parents):
        if (candidate / "intergrax").is_dir() and (candidate / "agents").is_dir():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            return candidate
    raise RuntimeError(
        f"Could not locate Intergrax repository root from {probe}. "
        "Open the notebook from the repo or set cwd to the repository root."
    )


class ExperimentRunOutcome(BaseModel):
    """Bundle returned after a Nexus experiment run."""

    record: ExperimentRecord
    task_result: TaskResult
    trace_event_count: int = 0
    checks: Dict[str, bool] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def evaluate_against_criteria(
    record: ExperimentRecord,
    result: TaskResult,
) -> Dict[str, bool]:
    """Apply lightweight checks from registry validation fields."""
    checks: Dict[str, bool] = {
        "completed": result.state == TaskState.COMPLETED,
        "validation_valid": result.metadata.get("validation_valid") is True,
        "non_empty_answer": bool(result.answer.strip()),
    }
    expected = record.expected_output.strip()
    if expected:
        checks["expected_output_substring"] = expected in result.answer
    return checks


class ExperimentSession:
    """
    Laboratory session: register hypothesis → run via Nexus → link trace → decide.

    Example (notebook or script)::

        session = ExperimentSession(trace_db=Path("build/notebook_trace.db"))
        record = session.register(RegisterExperimentRequest(...))
        loop = session.build_nexus_loop(build_harness_registry())
        outcome = await session.run(loop=loop, record=record, message="hello")
        session.decide(record.experiment_id, ExperimentDecision.KEEP)
    """

    def __init__(
        self,
        *,
        experiments_db: Path | None = None,
        trace_db: Path | None = None,
        tenant_id: str = "t1",
        user_id: str = "u1",
        auto_link_runs: bool = True,
    ) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.auto_link_runs = auto_link_runs
        self._experiment_store: SQLiteExperimentStore = open_experiment_store(experiments_db)
        self._trace_db = trace_db
        if trace_db is not None:
            trace_db.parent.mkdir(parents=True, exist_ok=True)

    @property
    def experiment_store(self) -> SQLiteExperimentStore:
        return self._experiment_store

    @property
    def trace_db(self) -> Path | None:
        return self._trace_db

    def register(self, request: RegisterExperimentRequest) -> ExperimentRecord:
        return self._experiment_store.register(request)

    def build_nexus_loop(self, registry: AgentRegistry) -> NexusLoop:
        trace_store = None
        if self._trace_db is not None:
            trace_store = SQLiteRunTraceStore(db_path=self._trace_db)
        return NexusLoop(registry, trace_store=trace_store)

    async def run(
        self,
        *,
        loop: NexusLoop,
        record: ExperimentRecord,
        message: str,
        capability: str | None = None,
    ) -> ExperimentRunOutcome:
        task = Task(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            message=message,
            context=TaskContext(
                capability=capability or record.capability,
            ),
        )
        if record.agent_id:
            task.agent_id = record.agent_id

        result = await loop.handle_task(task)
        run_id = result.run_id or result.task_id

        updated = record
        if self.auto_link_runs and run_id:
            updated = self._experiment_store.link_run(record.experiment_id, run_id)

        trace_count = 0
        if loop.trace_emitter is not None:
            trace_count = len(loop.trace_emitter.events)

        checks = evaluate_against_criteria(updated, result)
        return ExperimentRunOutcome(
            record=updated,
            task_result=result,
            trace_event_count=trace_count,
            checks=checks,
        )

    def link_run(self, experiment_id: str, run_id: str) -> ExperimentRecord:
        return self._experiment_store.link_run(experiment_id, run_id)

    def decide(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
        *,
        notes: Optional[str] = None,
    ) -> ExperimentRecord:
        return self._experiment_store.set_decision(experiment_id, decision, notes=notes)

    def summarize_trace(self, run_id: str) -> dict:
        """Read persisted trace metadata when ``trace_db`` was configured."""
        if self._trace_db is None or not self._trace_db.exists():
            raise FileNotFoundError(
                "Trace database not configured or missing. "
                "Pass trace_db= to ExperimentSession before running."
            )
        store = SQLiteRunTraceStore(db_path=self._trace_db)
        persisted = store.read_run(run_id, self.tenant_id)
        stats = persisted.metadata.stats if persisted.metadata else None
        llm_usage = dict(stats.llm_usage or {}) if stats else {}
        return {
            "run_id": run_id,
            "event_count": len(persisted.events),
            "duration_ms": stats.duration_ms if stats else None,
            "cost": llm_usage.get("cost"),
            "total_tokens": llm_usage.get("total_tokens"),
            "lifecycle_steps": [
                event.get("step")
                for event in persisted.events
                if event.get("step") == "task_lifecycle"
            ],
        }
