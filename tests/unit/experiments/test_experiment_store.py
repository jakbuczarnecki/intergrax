# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.experiments.models import ExperimentDecision, RegisterExperimentRequest
from intergrax.experiments.store import SQLiteExperimentStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture
def store(tmp_path):
    return SQLiteExperimentStore(db_path=tmp_path / "experiments.db")


def test_register_and_get_experiment(store: SQLiteExperimentStore):
    record = store.register(
        RegisterExperimentRequest(
            hypothesis="Echo responds deterministically",
            capability="echo.basic",
            agent_id="echo",
            expected_output="echo: hello",
            validation_criteria="non-empty prefixed answer",
        )
    )
    loaded = store.get(record.experiment_id)
    assert loaded.hypothesis == "Echo responds deterministically"
    assert loaded.capability == "echo.basic"
    assert loaded.decision == ExperimentDecision.PENDING


def test_link_run_and_decide(store: SQLiteExperimentStore):
    record = store.register(
        RegisterExperimentRequest(
            hypothesis="Legal review smoke",
            capability="legal.contract_review",
        )
    )
    linked = store.link_run(record.experiment_id, "run-legal-1")
    assert linked.run_ids == ["run-legal-1"]

    kept = store.set_decision(record.experiment_id, ExperimentDecision.KEEP, notes="good trace")
    assert kept.decision == ExperimentDecision.KEEP
    assert kept.notes == "good trace"


def test_delete_removes_experiment(store: SQLiteExperimentStore):
    record = store.register(
        RegisterExperimentRequest(
            hypothesis="Discard me",
            capability="research.pipeline",
        )
    )
    store.set_decision(record.experiment_id, ExperimentDecision.DELETE)
    with pytest.raises(ValueError, match="not found"):
        store.get(record.experiment_id)


def test_list_filters_by_decision(store: SQLiteExperimentStore):
    pending = store.register(
        RegisterExperimentRequest(hypothesis="pending one", capability="echo.basic")
    )
    other = store.register(
        RegisterExperimentRequest(hypothesis="keep one", capability="echo.basic")
    )
    store.set_decision(other.experiment_id, ExperimentDecision.KEEP)

    pending_rows = store.list_experiments(decision=ExperimentDecision.PENDING)
    assert any(row.experiment_id == pending.experiment_id for row in pending_rows)
    assert all(row.decision == ExperimentDecision.PENDING for row in pending_rows)
