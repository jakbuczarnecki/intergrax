# © Artur Czarnecki. All rights reserved.

"""PT-T01 NT-15 — SQLite experiment store tenant contract probe (owner T-IGX-043-J)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from intergrax.experiments.models import RegisterExperimentRequest
from intergrax.experiments.store import SQLiteExperimentStore

pytestmark = pytest.mark.unit


def test_nt15_experiment_store_tenant_api_not_implemented_on_train(tmp_path: Path) -> None:
    """T-IGX-065-A tracks NT-15; tenant isolation on the store is PT-WF (T-IGX-043-J), not PT-T01."""
    store = SQLiteExperimentStore(db_path=tmp_path / "experiments.db")
    store.register(RegisterExperimentRequest(hypothesis="a", capability="cap"))
    store.register(RegisterExperimentRequest(hypothesis="b", capability="cap"))
    assert len(store.list_experiments()) == 2
    assert "tenant_id" not in inspect.signature(store.list_experiments).parameters
    assert "tenant_id" not in inspect.signature(store.get).parameters
