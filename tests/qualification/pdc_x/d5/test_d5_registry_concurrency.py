# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R018 (DX-07): concurrent writes to the file registry are safe or rejected."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    OnlineEvaluationRegistryBusyError,
)

pytestmark = pytest.mark.unit


def _observation(index: int) -> OnlineEvaluationObservation:
    return OnlineEvaluationObservation(
        observation_id=f"obs-{index}",
        run_id=f"run-{index}",
        agent_id="agent:research",
        mode=OnlineEvaluationMode.ONLINE,
        scenario_id="s-math",
        passed=True,
        score=1.0,
    )


def test_r018_concurrent_writers_lose_no_observation(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    threads_count, per_thread = 8, 6
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        # One instance per worker: only the file lock can serialize them.
        registry = FileOnlineEvaluationRegistry(path)
        try:
            for index in range(per_thread):
                registry.append(_observation(offset * 100 + index))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    stored = FileOnlineEvaluationRegistry(path).list_observations()
    assert len(stored) == threads_count * per_thread
    assert len({item.observation_id for item in stored}) == threads_count * per_thread
    json.loads(path.read_text(encoding="utf-8"))  # never truncated or interleaved
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "obs.json.lock").exists()


def test_r018_writer_is_rejected_while_the_registry_is_locked(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    registry = FileOnlineEvaluationRegistry(path, lock_timeout_seconds=0.2)
    registry.append(_observation(1))
    (tmp_path / "obs.json.lock").write_text("held by another process", encoding="utf-8")
    with pytest.raises(OnlineEvaluationRegistryBusyError):
        registry.append(_observation(2))
    with pytest.raises(OnlineEvaluationRegistryBusyError):
        registry.list_observations()
    (tmp_path / "obs.json.lock").unlink()
    assert [item.observation_id for item in registry.list_observations()] == ["obs-1"]


def test_r018_stale_lock_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    lock = tmp_path / "obs.json.lock"
    lock.write_text("crashed writer", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    registry = FileOnlineEvaluationRegistry(path, lock_timeout_seconds=1.0, stale_lock_seconds=5.0)
    registry.append(_observation(1))
    assert len(registry.list_observations()) == 1


def test_r018_registry_file_is_replaced_atomically(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    registry = FileOnlineEvaluationRegistry(path)
    registry.append(_observation(1))
    before = path.read_text(encoding="utf-8")
    registry.append(_observation(2))
    after = path.read_text(encoding="utf-8")
    assert json.loads(before)["observations"][0]["observation_id"] == "obs-1"
    assert len(json.loads(after)["observations"]) == 2
    assert list(tmp_path.iterdir()) == [path]


def test_r018_legacy_registry_file_without_tenant_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    legacy = {
        "schema_version": "1.0.0",
        "observations": [json.loads(_observation(7).model_dump_json())],
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    (item,) = FileOnlineEvaluationRegistry(path).list_observations()
    assert item.observation_id == "obs-7"
