# © Artur Czarnecki. All rights reserved.

"""AW-4B — PostgreSQL goal evaluation cadence integration tests."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.autonomous_work.goal_evaluation_ports import MappingGoalEvaluationCadenceResolver
from intergrax.autonomous_work.goal_evaluation_service import WorkerGoalEvaluationService
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.postgresql_repository import PostgreSQLAutonomousWorkStore
from intergrax.contracts.autonomous_work import (
    GoalEvaluationDisposition,
    GoalProgressProjection,
    WorkerGoalEvaluationRequest,
    WorkerLifecycleState,
    mint_wake_up_id,
)
from intergrax.contracts.autonomous_work.references import (
    EvaluationCadenceRef,
    ProgressProjectionRef,
    WakeUpSourceRef,
)
from intergrax.contracts.autonomous_work.wake_up import (
    WorkerWakeUpContext,
    WorkerWakeUpDisposition,
    WorkerWakeUpSourceKind,
    WorkerWakeUpSignal,
)
from tests.integration.autonomous_work.conftest import open_bundle
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = [pytest.mark.integration, pytest.mark.network]

_UTC = UTC
_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=_UTC)
_CADENCE_REF = EvaluationCadenceRef("cadence/goal-eval-1h")
_CADENCE_SECONDS = 3600
_PROJECTION_REF = ProgressProjectionRef("projection/sla-30m")


@pytest.fixture
def cadence_repo(postgresql_autonomous_work_bundle: AutonomousWorkRepositories):
    return postgresql_autonomous_work_bundle.goal_evaluation_cadence_state


def test_postgresql_goal_evaluation_cadence_state_contracts(cadence_repo: object) -> None:
    contract_suite.contract_goal_evaluation_cadence_state_repository(cadence_repo)
    contract_suite.contract_goal_evaluation_cadence_state_goal_isolation(cadence_repo)


def test_postgresql_fresh_database_bootstraps_schema_v4(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
        ).fetchone()
        assert row is not None
        assert int(row["schema_version"]) == 4
        table_row = conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'aw_goal_evaluation_cadence_states'
            """
        ).fetchone()
        assert table_row is not None


def test_postgresql_v3_to_v4_migration_preserves_existing_data(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS aw_goal_evaluation_cadence_states")
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (3,),
        )
    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded == created
        with migrated_bundle.store.transaction() as conn:
            row = conn.execute(
                "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
            ).fetchone()
            assert row is not None
            assert int(row["schema_version"]) == 4
    finally:
        migrated_bundle.close()


def test_postgresql_restart_cadence_eligibility(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker = contract_suite.worker_instance()
    worker_id = worker.worker_instance_id
    postgresql_autonomous_work_bundle.worker_instance.create(worker)
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    postgresql_autonomous_work_bundle.responsibility.create(responsibility)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        evaluation_cadence_ref=_CADENCE_REF,
        progress_projection_ref=_PROJECTION_REF,
    )
    postgresql_autonomous_work_bundle.worker_goal.create(goal)
    cadence_repo = postgresql_autonomous_work_bundle.goal_evaluation_cadence_state
    service = WorkerGoalEvaluationService(
        responsibility_repository=postgresql_autonomous_work_bundle.responsibility,
        worker_goal_repository=postgresql_autonomous_work_bundle.worker_goal,
        cadence_resolver=MappingGoalEvaluationCadenceResolver(
            {_CADENCE_REF: _CADENCE_SECONDS}
        ),
        progress_projection_resolver=_projection_resolver(),
        goal_evaluator=__import__(
            "intergrax.autonomous_work.goal_evaluation_ports",
            fromlist=["DeterministicThresholdGoalEvaluator"],
        ).DeterministicThresholdGoalEvaluator(),
        cadence_state=cadence_repo,
    )
    first = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert first.goals_evaluated == 1
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    postgresql_autonomous_work_bundle.close()
    restarted = open_bundle(schema_name)
    try:
        restarted_service = WorkerGoalEvaluationService(
            responsibility_repository=restarted.responsibility,
            worker_goal_repository=restarted.worker_goal,
            cadence_resolver=MappingGoalEvaluationCadenceResolver(
                {_CADENCE_REF: _CADENCE_SECONDS}
            ),
            progress_projection_resolver=_projection_resolver(),
            goal_evaluator=__import__(
                "intergrax.autonomous_work.goal_evaluation_ports",
                fromlist=["DeterministicThresholdGoalEvaluator"],
            ).DeterministicThresholdGoalEvaluator(),
            cadence_state=restarted.goal_evaluation_cadence_state,
        )
        not_due = restarted_service.evaluate(
            WorkerGoalEvaluationRequest(
                wake_up_context=_accepted_context(worker_id=worker_id),
                evaluated_at=_NOW + timedelta(minutes=30),
                max_goals=10,
            )
        )
        assert not_due.decisions[0].disposition == GoalEvaluationDisposition.NOT_DUE
        due = restarted_service.evaluate(
            WorkerGoalEvaluationRequest(
                wake_up_context=_accepted_context(worker_id=worker_id),
                evaluated_at=_NOW + timedelta(hours=1),
                max_goals=10,
            )
        )
        assert due.decisions[0].disposition != GoalEvaluationDisposition.NOT_DUE
    finally:
        restarted.close()


def test_postgresql_concurrent_cadence_record_is_monotonic(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    goal_id = contract_suite.worker_goal().goal_id
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    barrier = threading.Barrier(2)
    results: list[datetime] = []
    errors: list[BaseException] = []

    def attempt(offset_minutes: int) -> None:
        bundle = open_bundle(schema_name)
        try:
            barrier.wait(timeout=5)
            state = bundle.goal_evaluation_cadence_state.record_evaluated(
                goal_id=goal_id,
                evaluated_at=_NOW + timedelta(minutes=offset_minutes),
            )
            results.append(state.last_evaluated_at)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [
        threading.Thread(target=attempt, args=(0,)),
        threading.Thread(target=attempt, args=(10,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    final = postgresql_autonomous_work_bundle.goal_evaluation_cadence_state.get(
        goal_id=goal_id
    )
    assert final is not None
    assert final.last_evaluated_at == max(results)
    assert final.last_evaluated_at >= _NOW


def _projection_resolver():
    from intergrax.autonomous_work.goal_evaluation_ports import (
        MappingGoalProgressProjectionResolver,
    )

    return MappingGoalProgressProjectionResolver(
        {
            _PROJECTION_REF: GoalProgressProjection(
                projection_ref=_PROJECTION_REF,
                observed_at=_NOW,
                current_value=1.0,
                target_value=1.0,
                status="healthy",
                evidence_refs=("evidence/progress/healthy",),
            )
        }
    )


def _accepted_context(*, worker_id: str) -> WorkerWakeUpContext:
    worker = contract_suite.worker_instance(
        worker_instance_id=worker_id,
        lifecycle_state=WorkerLifecycleState.IDLE,
    )
    wake_up_id = mint_wake_up_id()
    signal = WorkerWakeUpSignal(
        wake_up_id=wake_up_id,
        worker_instance_id=worker_id,
        source_kind=WorkerWakeUpSourceKind.SCHEDULE,
        source_ref=WakeUpSourceRef("schedule/goal-eval"),
        occurred_at=_NOW - timedelta(minutes=1),
        delivery_identity=wake_up_id,
    )
    return WorkerWakeUpContext(
        worker_instance=worker,
        wake_up_signal=signal,
        continuity_state=None,
        accepted_at=_NOW,
        disposition=WorkerWakeUpDisposition.ACCEPTED,
        receipt=None,
    )

