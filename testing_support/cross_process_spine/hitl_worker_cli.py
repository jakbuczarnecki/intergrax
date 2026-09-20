# © Artur Czarnecki. All rights reserved.

"""Subprocess HITL pause/resume entry for OBS-DIAG-X4."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from intergrax.contracts.execution_identity import RunId
from intergrax.runtime.diagnostics.persistence_conformance import query_all_problems_for_tenant
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
    restore_durable_continuation_backing,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.task.task import TaskState
from testing_support.cross_process_spine.durable_document_store import SqliteFileDocumentStore
from testing_support.cross_process_spine.models import HitlProcessResult, HitlSpineScenarioConfig
from testing_support.obs_universal_spine.hitl_restart_harness import (
    ObsSpineHitlAgent,
    _pause_task,
    _prepare_hitl_resume_task,
    _resume_task,
    build_hitl_restart_runtime,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
)


def _document_stack(work_dir: Path) -> tuple[SqliteFileDocumentStore, object, object]:
    document_store = SqliteFileDocumentStore(
        work_dir / "problems.docstore.sqlite",
        cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET,
    )
    return (
        document_store,
        document_store_problem_persistence_for_tests(document_store),
        document_store_occurrence_persistence_for_tests(document_store),
    )


async def _run_pause(config: HitlSpineScenarioConfig) -> HitlProcessResult:
    work_dir = Path(config.work_dir)
    document_store, problem_persistence, occurrence_persistence = _document_stack(work_dir)
    continuation_backing = ExecutionContinuationDurableBacking()
    runtime = build_hitl_restart_runtime(
        checkpoint_db=Path(config.checkpoint_db_path),
        runtime_events_db=Path(config.runtime_events_db_path),
        continuation_backing=continuation_backing,
        document_store=document_store,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
    )
    ObsSpineHitlAgent.step_run_count = 0
    run_id = RunId(config.run_id)
    paused = await runtime.runner.run_task(_pause_task(tenant_id=config.tenant_id), run_id=run_id)
    assert paused.state is TaskState.WAITING_FOR_HUMAN
    assert paused.summary.resume_token
    loaded = runtime.checkpoint_store.get_latest(paused.task_id, config.tenant_id)
    assert loaded is not None and loaded.runtime is not None
    Path(config.continuation_export_path).write_text(
        json.dumps(
            export_durable_continuation_state(continuation_backing),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    problems = query_all_problems_for_tenant(runtime.read_deps.problem_persistence, config.tenant_id)
    return HitlProcessResult(
        process_pid=os.getpid(),
        parent_pid=os.getppid(),
        phase="pause",
        tenant_id=config.tenant_id,
        task_id=paused.task_id,
        run_id=str(run_id),
        attempt_id=str(loaded.runtime.attempt_id),
        execution_id=str(loaded.runtime.execution_tree.entries[0].execution_id),
        pause_state=paused.state.value,
        resume_token=paused.summary.resume_token,
        problem_count=len(problems),
        problem_ids=tuple(problem.problem_id for problem in problems),
    )


async def _run_resume(config: HitlSpineScenarioConfig) -> HitlProcessResult:
    if not config.task_id or not config.resume_token:
        raise ValueError("resume phase requires task_id and resume_token")
    work_dir = Path(config.work_dir)
    document_store, problem_persistence, occurrence_persistence = _document_stack(work_dir)
    export_payload = json.loads(
        Path(config.continuation_export_path).read_text(encoding="utf-8"),
    )
    backing = restore_durable_continuation_backing(export_payload)
    continuation_store = execution_continuation_state_store_from_durable_export(export_payload)
    runtime = build_hitl_restart_runtime(
        checkpoint_db=Path(config.checkpoint_db_path),
        runtime_events_db=Path(config.runtime_events_db_path),
        inject_violation=config.inject_violation_on_resume,
        continuation_backing=backing,
        execution_continuation_state_store=continuation_store,
        document_store=document_store,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
    )
    run_id = RunId(config.run_id)
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=Path(config.checkpoint_db_path))
    loaded = checkpoint_store.get_latest(config.task_id, config.tenant_id)
    assert loaded is not None and loaded.runtime is not None
    resume_task = _resume_task(
        task_id=config.task_id,
        resume_token=config.resume_token,
        human_approved=config.human_approved,
        human_rejected=config.human_rejected,
        tenant_id=config.tenant_id,
    )
    _prepare_hitl_resume_task(
        resume_task,
        loaded=loaded,
        run_id=run_id,
        human_approved=config.human_approved,
        human_rejected=config.human_rejected,
    )
    resumed = await runtime.runner.run_task(
        resume_task,
        run_id=run_id,
        attempt_id=loaded.runtime.attempt_id,
        resume_checkpoint=loaded,
    )
    latest = runtime.checkpoint_store.get_latest(config.task_id, config.tenant_id)
    assert latest is not None and latest.runtime is not None
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime.runtime_event_store,
        causal_evidence=runtime.causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(
        config.tenant_id,
        resumed.task_id,
        run_id,
    )
    problems = query_all_problems_for_tenant(runtime.read_deps.problem_persistence, config.tenant_id)
    return HitlProcessResult(
        process_pid=os.getpid(),
        parent_pid=os.getppid(),
        phase="resume",
        tenant_id=config.tenant_id,
        task_id=resumed.task_id,
        run_id=str(run_id),
        attempt_id=str(latest.runtime.attempt_id),
        execution_id=str(latest.runtime.execution_tree.entries[0].execution_id),
        terminal_state=resumed.state.value,
        problem_count=len(problems),
        problem_ids=tuple(problem.problem_id for problem in problems),
        reconstruction_has_events=reconstruction.has_runtime_events,
        reconstruction_complete=reconstruction.is_runtime_history_complete,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OBS-DIAG-X4 HITL subprocess")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args(argv)
    config = HitlSpineScenarioConfig.model_validate_json(
        args.config_path.read_text(encoding="utf-8"),
    )
    result = asyncio.run(_run_pause(config) if config.phase == "pause" else _run_resume(config))
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
