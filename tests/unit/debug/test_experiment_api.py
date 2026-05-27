# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.debug.app import create_debug_app
from intergrax.experiments.models import ExperimentDecision

pytestmark = pytest.mark.unit


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "trace.db"
    db_path.touch()
    experiments_db = tmp_path / "experiments.db"
    app = create_debug_app(db_path=db_path, experiments_db_path=experiments_db)
    with TestClient(app) as test_client:
        yield test_client


def test_debug_api_register_and_list_experiments(client: TestClient):
    response = client.post(
        "/debug/experiments",
        json={
            "hypothesis": "Research pipeline produces summary",
            "capability": "research.pipeline",
            "agent_id": "research",
            "expected_output": "summary prefix",
            "validation_criteria": "two-agent graph completes",
        },
    )
    assert response.status_code == 201, response.text
    experiment_id = response.json()["experiment_id"]

    listed = client.get("/debug/experiments")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    detail = client.get(f"/debug/experiments/{experiment_id}")
    assert detail.status_code == 200
    assert detail.json()["hypothesis"].startswith("Research pipeline")


def test_debug_api_decide_and_link_run(client: TestClient):
    created = client.post(
        "/debug/experiments",
        json={
            "hypothesis": "Echo gate path",
            "capability": "echo.basic",
        },
    ).json()
    experiment_id = created["experiment_id"]

    linked = client.post(f"/debug/experiments/{experiment_id}/runs/run-echo-1")
    assert linked.status_code == 200
    assert "run-echo-1" in linked.json()["run_ids"]

    decided = client.post(
        f"/debug/experiments/{experiment_id}/decision",
        json={"decision": ExperimentDecision.IMPROVE.value, "notes": "add cost metric"},
    )
    assert decided.status_code == 200
    assert decided.json()["decision"] == ExperimentDecision.IMPROVE.value


def test_debug_api_delete_experiment(client: TestClient):
    created = client.post(
        "/debug/experiments",
        json={"hypothesis": "temporary", "capability": "echo.basic"},
    ).json()
    experiment_id = created["experiment_id"]

    deleted = client.post(
        f"/debug/experiments/{experiment_id}/decision",
        json={"decision": ExperimentDecision.DELETE.value},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    missing = client.get(f"/debug/experiments/{experiment_id}")
    assert missing.status_code == 404
