# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.experiments.composition import resolve_experiment_persistence
from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
)
from intergrax.experiments.persistence_contract import ExperimentPersistence
from intergrax.experiments.store import SQLiteExperimentStore
from intergrax.experiments.workflow import ExperimentSession
from intergrax.integrations.providers.relational_store.sqlite import (
    create_sqlite_experiment_store,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class FakeExperimentStore:
    """In-memory experiment persistence for provider-substitution proofs."""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}
        self.register_calls: list[RegisterExperimentRequest] = []
        self.link_calls: list[tuple[str, str]] = []
        self.decision_calls: list[tuple[str, ExperimentDecision, str | None]] = []

    def register(self, request: RegisterExperimentRequest) -> ExperimentRecord:
        self.register_calls.append(request)
        record = ExperimentRecord.new_from_request(
            request,
            created_at_utc="2026-01-01T00:00:00Z",
            updated_at_utc="2026-01-01T00:00:00Z",
        )
        self._records[record.experiment_id] = record
        return record

    def list_experiments(
        self,
        *,
        limit: int = 50,
        decision: ExperimentDecision | None = None,
    ) -> list[ExperimentRecord]:
        rows = list(self._records.values())
        if decision is not None:
            rows = [row for row in rows if row.decision == decision]
        return rows[:limit]

    def get(self, experiment_id: str) -> ExperimentRecord:
        record = self._records.get(experiment_id)
        if record is None:
            raise ValueError(f"Experiment not found: {experiment_id}")
        return record

    def set_decision(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
        *,
        notes: str | None = None,
    ) -> ExperimentRecord:
        self.decision_calls.append((experiment_id, decision, notes))
        record = self.get(experiment_id)
        updated = record.model_copy(
            update={
                "decision": decision,
                "notes": notes or record.notes,
                "updated_at_utc": "2026-01-02T00:00:00Z",
            }
        )
        self._records[experiment_id] = updated
        return updated

    def link_run(self, experiment_id: str, run_id: str) -> ExperimentRecord:
        self.link_calls.append((experiment_id, run_id))
        record = self.get(experiment_id)
        run_ids = list(record.run_ids)
        if run_id not in run_ids:
            run_ids.append(run_id)
        updated = record.model_copy(
            update={
                "run_ids": run_ids,
                "updated_at_utc": "2026-01-02T00:00:00Z",
            }
        )
        self._records[experiment_id] = updated
        return updated


def test_sqlite_store_satisfies_experiment_persistence(tmp_path: Path):
    store = SQLiteExperimentStore(db_path=tmp_path / "experiments.db")
    assert isinstance(store, ExperimentPersistence)


def test_create_sqlite_experiment_store_satisfies_port(tmp_path: Path):
    store = create_sqlite_experiment_store(db_path=tmp_path / "experiments.db")
    assert isinstance(store, ExperimentPersistence)


def test_experiment_session_uses_fake_store_without_sqlite():
    fake = FakeExperimentStore()
    session = ExperimentSession(experiment_store=fake)

    record = session.register(
        RegisterExperimentRequest(
            hypothesis="Fake store proof",
            capability="echo.basic",
        )
    )
    assert fake.register_calls
    assert record.experiment_id in fake._records

    linked = session.link_run(record.experiment_id, "run-fake-1")
    assert fake.link_calls == [(record.experiment_id, "run-fake-1")]
    assert linked.run_ids == ["run-fake-1"]

    decided = session.decide(
        record.experiment_id,
        ExperimentDecision.KEEP,
        notes="port proof",
    )
    assert fake.decision_calls == [(record.experiment_id, ExperimentDecision.KEEP, "port proof")]
    assert decided.decision == ExperimentDecision.KEEP


def test_experiment_session_public_store_is_port():
    fake = FakeExperimentStore()
    session = ExperimentSession(experiment_store=fake)
    assert isinstance(session.experiment_store, ExperimentPersistence)


def test_workflow_has_no_sqlite_experiment_imports():
    source = Path("intergrax/experiments/workflow.py").read_text(encoding="utf-8")
    assert "SQLiteExperimentStore" not in source
    assert "create_sqlite_experiment_store" not in source


def test_default_local_dx_still_uses_sqlite(tmp_path: Path):
    experiments_db = tmp_path / "experiments.db"
    session = ExperimentSession(experiments_db=experiments_db)
    assert isinstance(session.experiment_store, SQLiteExperimentStore)
    assert experiments_db.exists()


def test_resolve_experiment_persistence_prefers_injected_store(tmp_path: Path):
    fake = FakeExperimentStore()
    resolved = resolve_experiment_persistence(
        experiment_store=fake,
        experiments_db=tmp_path / "unused.db",
    )
    assert resolved is fake
