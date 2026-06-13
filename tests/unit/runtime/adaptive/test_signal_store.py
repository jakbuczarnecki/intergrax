# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.3: Signal store persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore, SQLiteSignalStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _sample_signal(*, tenant_id: str = "t1", run_id: str = "run_1") -> HarnessOutcomeSignal:
    return HarnessOutcomeSignal(
        run_id=run_id,
        tenant_id=tenant_id,
        application_id="lab.default",
        agent_id="echo",
        task_class="echo.basic",
        utility=0.75,
    )


def test_in_memory_signal_store_filters_by_tenant() -> None:
    store = InMemorySignalStore()
    store.append(_sample_signal(tenant_id="t1", run_id="run_a"))
    store.append(_sample_signal(tenant_id="t2", run_id="run_b"))
    assert len(store.list_signals(tenant_id="t1")) == 1
    assert store.list_signals(tenant_id="t1")[0].run_id == "run_a"


def test_sqlite_signal_store_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    store = SQLiteSignalStore(db_path=db_path)
    signal = _sample_signal()
    store.append(signal)
    loaded = store.list_signals(limit=10)
    assert len(loaded) == 1
    assert loaded[0].signal_id == signal.signal_id
    assert loaded[0].utility == 0.75


def test_sqlite_signal_store_time_window(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    store = SQLiteSignalStore(db_path=db_path)
    now = datetime.now(UTC)
    older = _sample_signal(run_id="run_old").model_copy(update={"timestamp": now - timedelta(hours=2)})
    newer = _sample_signal(run_id="run_new").model_copy(update={"timestamp": now})
    store.append(older)
    store.append(newer)
    recent = store.list_signals(since=now - timedelta(hours=1), limit=10)
    assert len(recent) == 1
    assert recent[0].run_id == "run_new"
