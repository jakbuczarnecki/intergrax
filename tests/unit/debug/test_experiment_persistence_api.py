# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intergrax.debug.router import create_debug_router
from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
)
from intergrax.experiments.persistence_contract import ExperimentPersistence

pytestmark = pytest.mark.unit


class FakeExperimentStore:
    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}

    def register(self, request: RegisterExperimentRequest) -> ExperimentRecord:
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
        record = self.get(experiment_id)
        if decision == ExperimentDecision.DELETE:
            del self._records[experiment_id]
            return record.model_copy(update={"decision": decision, "notes": notes or ""})
        updated = record.model_copy(
            update={
                "decision": decision,
                "notes": notes or record.notes,
            }
        )
        self._records[experiment_id] = updated
        return updated

    def link_run(self, experiment_id: str, run_id: str) -> ExperimentRecord:
        record = self.get(experiment_id)
        run_ids = list(record.run_ids)
        if run_id not in run_ids:
            run_ids.append(run_id)
        updated = record.model_copy(update={"run_ids": run_ids})
        self._records[experiment_id] = updated
        return updated


@pytest.fixture
def fake_client():
    fake = FakeExperimentStore()
    router = create_debug_router(experiment_store=fake)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, fake


def test_debug_api_uses_fake_experiment_store(fake_client):
    client, fake = fake_client

    created = client.post(
        "/debug/experiments",
        json={
            "hypothesis": "Fake API proof",
            "capability": "echo.basic",
        },
    )
    assert created.status_code == 201, created.text
    experiment_id = created.json()["experiment_id"]
    assert experiment_id in fake._records

    listed = client.get("/debug/experiments")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    detail = client.get(f"/debug/experiments/{experiment_id}")
    assert detail.status_code == 200
    assert detail.json()["hypothesis"] == "Fake API proof"

    linked = client.post(f"/debug/experiments/{experiment_id}/runs/run-fake-api")
    assert linked.status_code == 200
    assert "run-fake-api" in linked.json()["run_ids"]

    decided = client.post(
        f"/debug/experiments/{experiment_id}/decision",
        json={"decision": ExperimentDecision.KEEP.value, "notes": "fake"},
    )
    assert decided.status_code == 200
    assert decided.json()["decision"] == ExperimentDecision.KEEP.value


def test_debug_router_has_no_sqlite_experiment_imports():
    source = Path("intergrax/debug/router.py").read_text(encoding="utf-8")
    assert "SQLiteExperimentStore" not in source
    assert "open_experiment_store" not in source
    assert "ExperimentPersistence = Depends(get_experiments)" in source
