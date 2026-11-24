# © Artur Czarnecki. All rights reserved.

import pytest
from intergrax.experiments.models import ExperimentDecision, RegisterExperimentRequest
from intergrax.experiments.workflow import (
    ExperimentSession,
    ensure_repo_root_on_path,
    evaluate_against_criteria,
)
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from testing_support.agent_registry_bootstrap import build_harness_registry

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_ensure_repo_root_on_path(tmp_path):
    (tmp_path / "intergrax").mkdir()
    (tmp_path / "agents").mkdir()
    root = ensure_repo_root_on_path(tmp_path)
    assert root == tmp_path.resolve()


@pytest.mark.asyncio
async def test_experiment_session_echo_workflow(tmp_path):
    experiments_db = tmp_path / "experiments.db"
    trace_db = tmp_path / "trace.db"

    session = ExperimentSession(
        experiments_db=experiments_db,
        trace_db=trace_db,
        tenant_id="t1",
        user_id="u1",
    )
    record = session.register(
        RegisterExperimentRequest(
            hypothesis="Echo returns prefixed answer",
            capability="echo.basic",
            agent_id="echo",
            expected_output="hello workflow",
            validation_criteria="non-empty answer with echo prefix",
        )
    )

    loop = session.build_nexus_loop(build_harness_registry())
    outcome = await session.run(
        loop=loop,
        record=record,
        message="hello workflow",
    )

    assert outcome.passed
    assert outcome.task_result.state == TaskState.COMPLETED
    assert "hello workflow" in outcome.task_result.answer
    assert outcome.record.run_ids
    assert outcome.trace_event_count > 0

    summary = session.summarize_trace(outcome.task_result.run_id or outcome.task_result.task_id)
    assert summary["event_count"] > 0

    decided = session.decide(
        record.experiment_id,
        ExperimentDecision.KEEP,
        notes="gate test",
    )
    assert decided.decision == ExperimentDecision.KEEP


@pytest.fixture
def two_tenant_experiment_store(tmp_path):
    from intergrax.experiments.store import SQLiteExperimentStore

    db = tmp_path / "experiments.db"
    store = SQLiteExperimentStore(db_path=db)
    record_a = store.register(
        RegisterExperimentRequest(
            tenant_id="tenant-a",
            hypothesis="hypothesis-a",
            capability="echo.basic",
            agent_id="echo",
        )
    )
    record_b = store.register(
        RegisterExperimentRequest(
            tenant_id="tenant-b",
            hypothesis="hypothesis-b",
            capability="echo.basic",
            agent_id="echo",
        )
    )
    return store, record_a, record_b


def test_wf_session_registration_binds_implicit_tenant(two_tenant_experiment_store) -> None:
    store, _, _ = two_tenant_experiment_store
    session = ExperimentSession(experiment_store=store, tenant_id="tenant-a")
    record = session.register(
        RegisterExperimentRequest(
            hypothesis="implicit tenant",
            capability="echo.basic",
        )
    )
    assert record.tenant_id == "tenant-a"


def test_wf_session_registration_rejects_foreign_tenant(two_tenant_experiment_store) -> None:
    store, _, _ = two_tenant_experiment_store
    session = ExperimentSession(experiment_store=store, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="not authorized"):
        session.register(
            RegisterExperimentRequest(
                tenant_id="tenant-b",
                hypothesis="foreign",
                capability="echo.basic",
            )
        )


def test_wf_session_decide_rejects_foreign_experiment(two_tenant_experiment_store) -> None:
    store, _, record_b = two_tenant_experiment_store
    session = ExperimentSession(experiment_store=store, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="not found"):
        session.decide(record_b.experiment_id, ExperimentDecision.KEEP)


def test_wf_session_link_run_rejects_foreign_experiment(two_tenant_experiment_store) -> None:
    store, _, record_b = two_tenant_experiment_store
    session = ExperimentSession(experiment_store=store, tenant_id="tenant-a")
    with pytest.raises(ValueError, match="not found"):
        session.link_run(record_b.experiment_id, "run-foreign")
    unchanged = store.get(record_b.experiment_id, tenant_id="tenant-b")
    assert unchanged.run_ids == []


@pytest.mark.asyncio
async def test_wf_session_run_rejects_foreign_record_before_execution(
    two_tenant_experiment_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    store, _, record_b = two_tenant_experiment_store
    session = ExperimentSession(
        experiment_store=store,
        trace_db=tmp_path / "trace.db",
        tenant_id="tenant-a",
    )
    loop = session.build_nexus_loop(build_harness_registry())
    run_called = {"count": 0}
    original = UnifiedTaskRunner.run_task

    async def guarded(self, *args, **kwargs):
        run_called["count"] += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UnifiedTaskRunner, "run_task", guarded)
    with pytest.raises(ValueError, match="not authorized"):
        await session.run(loop=loop, record=record_b, message="hello")
    assert run_called["count"] == 0


@pytest.mark.asyncio
async def test_wf_session_run_rejects_forged_record_identity(
    two_tenant_experiment_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    store, record_a, _ = two_tenant_experiment_store
    forged = record_a.model_copy(update={"capability": "forged.capability"})
    session = ExperimentSession(
        experiment_store=store,
        trace_db=tmp_path / "trace.db",
        tenant_id="tenant-a",
    )
    loop = session.build_nexus_loop(build_harness_registry())
    run_called = {"count": 0}
    original = UnifiedTaskRunner.run_task

    async def guarded(self, *args, **kwargs):
        run_called["count"] += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UnifiedTaskRunner, "run_task", guarded)
    with pytest.raises(ValueError, match="persisted experiment identity"):
        await session.run(loop=loop, record=forged, message="hello")
    assert run_called["count"] == 0


@pytest.mark.asyncio
async def test_wf_session_run_uses_authoritative_persisted_record(
    two_tenant_experiment_store,
    tmp_path,
) -> None:

    store, record_a, _ = two_tenant_experiment_store
    caller_record = record_a.model_copy(update={"agent_id": None})
    session = ExperimentSession(
        experiment_store=store,
        trace_db=tmp_path / "trace.db",
        tenant_id="tenant-a",
    )
    loop = session.build_nexus_loop(build_harness_registry())
    outcome = await session.run(
        loop=loop,
        record=caller_record,
        message="hello authoritative",
    )
    assert outcome.passed
    assert outcome.record.agent_id == "echo"


def test_wf_session_matching_tenant_preserves_workflow(two_tenant_experiment_store, tmp_path) -> None:
    store, record_a, _ = two_tenant_experiment_store
    session = ExperimentSession(
        experiment_store=store,
        experiments_db=tmp_path / "unused.db",
        tenant_id="tenant-a",
    )
    explicit = session.register(
        RegisterExperimentRequest(
            tenant_id="tenant-a",
            hypothesis="explicit ok",
            capability="echo.basic",
        )
    )
    assert explicit.tenant_id == "tenant-a"
    linked = session.link_run(record_a.experiment_id, "run-a-1")
    assert "run-a-1" in linked.run_ids
    decided = session.decide(record_a.experiment_id, ExperimentDecision.KEEP, notes="ok")
    assert decided.decision == ExperimentDecision.KEEP


def test_evaluate_against_criteria_expected_output():
    from intergrax.experiments.models import ExperimentRecord

    record = ExperimentRecord(
        experiment_id="exp1",
        hypothesis="test",
        capability="echo.basic",
        expected_output="needle",
        created_at_utc="2026-01-01T00:00:00Z",
        updated_at_utc="2026-01-01T00:00:00Z",
    )
    result = TaskResult(
        task_id="task1",
        run_id="run1",
        state=TaskState.COMPLETED,
        answer="echo: needle in haystack",
        metadata={"validation_valid": True},
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )
    checks = evaluate_against_criteria(record, result)
    assert checks["expected_output_substring"] is True
    assert checks["completed"] is True
