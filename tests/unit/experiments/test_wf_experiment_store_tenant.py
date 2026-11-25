# © Artur Czarnecki. All rights reserved.

"""PT-WF E4 — SQLite experiment registry tenant filtering."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.experiments.models import RegisterExperimentRequest
from intergrax.experiments.store import SQLiteExperimentStore

pytestmark = pytest.mark.unit


def test_experiment_store_scopes_list_and_get_by_tenant(tmp_path: Path) -> None:
    store = SQLiteExperimentStore(db_path=tmp_path / "experiments.db")
    record_a = store.register(
        RegisterExperimentRequest(
            tenant_id="tenant-a",
            hypothesis="a",
            capability="cap",
        )
    )
    store.register(
        RegisterExperimentRequest(
            tenant_id="tenant-b",
            hypothesis="b",
            capability="cap",
        )
    )
    listed_b = store.list_experiments(tenant_id="tenant-b")
    assert all(row.tenant_id == "tenant-b" for row in listed_b)
    assert record_a.experiment_id not in {row.experiment_id for row in listed_b}
    with pytest.raises(ValueError, match="not found"):
        store.get(record_a.experiment_id, tenant_id="tenant-b")
