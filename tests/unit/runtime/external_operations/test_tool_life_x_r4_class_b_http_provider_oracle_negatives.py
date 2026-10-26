# © Artur Czarnecki. All rights reserved.

"""Negative / concurrency / durability matrix for Class B HTTP qualification oracle."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.unit.runtime.external_operations.tool_life_x_r4_class_b_http_provider_oracle import (
    ClassBHttpProviderOracle,
    ClassBHttpProviderOracleConflictError,
    ClassBHttpProviderOraclePersistenceError,
    ClassBHttpProviderOracleRetentionExceededError,
    class_b_http_handler_for_oracle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BODY = b'{"action":"charge"}'
_ALT_BODY = b'{"action":"other"}'
_QUAL_INSERT_FAULT_TRIGGER = "qual_fault_class_b_effects_insert"


def _install_insert_fault_trigger(db: Path, rejected_key: str) -> None:
    escaped = rejected_key.replace("'", "''")
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"""
            CREATE TRIGGER {_QUAL_INSERT_FAULT_TRIGGER}
            BEFORE INSERT ON class_b_effects
            WHEN NEW.idempotency_key = '{escaped}'
            BEGIN
                SELECT RAISE(ABORT, 'qual_injected_sqlite_write_failure');
            END;
            """
        )
        conn.commit()


def _remove_insert_fault_trigger(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(f"DROP TRIGGER IF EXISTS {_QUAL_INSERT_FAULT_TRIGGER}")
        conn.commit()


def _row_exists(db: Path, idempotency_key: str) -> bool:
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT 1 FROM class_b_effects WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    return row is not None


def _apply_post_in_subprocess(db_path: str, key: str, body: bytes) -> None:
    oracle = ClassBHttpProviderOracle(db_path=Path(db_path))
    oracle.apply_post(idempotency_key=key, body=body)


def test_oracle_two_instances_shared_sqlite_single_effect(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    body = _BODY
    errors: list[BaseException] = []

    def _worker(instance_id: int) -> None:
        try:
            oracle = ClassBHttpProviderOracle(db_path=db)
            oracle.apply_post(idempotency_key="shared-k", body=body)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(i,))
        for i in range(2)
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
    assert not errors
    verifier = ClassBHttpProviderOracle(db_path=db)
    assert verifier.effect_count("shared-k") == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM class_b_effects").fetchone()[0] == 1


def test_oracle_concurrent_same_key_different_payload_one_conflict(tmp_path: Path) -> None:
    db = tmp_path / "conflict.db"
    results: list[str] = []
    lock = threading.Lock()

    def _worker(body: bytes) -> None:
        oracle = ClassBHttpProviderOracle(db_path=db)
        try:
            oracle.apply_post(idempotency_key="race-k", body=body)
            with lock:
                results.append("ok")
        except ClassBHttpProviderOracleConflictError:
            with lock:
                results.append("conflict")

    threads = [
        threading.Thread(target=_worker, args=(_BODY,)),
        threading.Thread(target=_worker, args=(_ALT_BODY,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert "conflict" in results
    assert results.count("ok") == 1
    assert ClassBHttpProviderOracle(db_path=db).effect_count("race-k") == 1


def test_oracle_subprocess_replay_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "proc.db"
    key = "proc-replay"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_apply_post_in_subprocess,
        args=(str(db), key, _BODY),
    )
    proc.start()
    proc.join(timeout=30.0)
    assert proc.exitcode == 0
    oracle = ClassBHttpProviderOracle(db_path=db)
    replay = oracle.apply_post(idempotency_key=key, body=_BODY)
    assert replay.replay and replay.effect_count == 1
    assert oracle.effect_count(key) == 1


def test_oracle_sqlite_insert_failure_rollback_no_phantom_cache(tmp_path: Path) -> None:
    """Deterministic SQLite write failure (INSERT ABORT), not COMMIT-time injection."""
    db = tmp_path / "fail.db"
    rejected_key = "new-key"
    oracle = ClassBHttpProviderOracle(db_path=db)
    seed = oracle.apply_post(idempotency_key="seed", body=_BODY)
    assert seed.replay is False and seed.effect_count == 1
    _install_insert_fault_trigger(db, rejected_key)
    try:
        with pytest.raises(ClassBHttpProviderOraclePersistenceError) as exc_info:
            oracle.apply_post(idempotency_key=rejected_key, body=_BODY)
        assert isinstance(exc_info.value.__cause__, sqlite3.Error)
        assert oracle.effect_count(rejected_key) == 0
        assert not _row_exists(db, rejected_key)
        assert rejected_key not in oracle._memory  # noqa: SLF001
        assert oracle.effect_count("seed") == 1
        assert _row_exists(db, "seed")
    finally:
        _remove_insert_fault_trigger(db)

    recovered = oracle.apply_post(idempotency_key=rejected_key, body=_BODY)
    assert recovered.replay is False and recovered.effect_count == 1
    replay = oracle.apply_post(idempotency_key=rejected_key, body=_BODY)
    assert replay.replay is True and replay.effect_count == 1
    assert oracle.effect_count(rejected_key) == 1

    fresh = ClassBHttpProviderOracle(db_path=db)
    assert fresh.effect_count(rejected_key) == 1
    assert fresh.effect_count("seed") == 1
    durable_replay = fresh.apply_post(idempotency_key=rejected_key, body=_BODY)
    assert durable_replay.replay is True
    assert fresh.effect_count(rejected_key) == 1


def test_oracle_corrupt_row_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "corrupt.db"
    oracle = ClassBHttpProviderOracle(db_path=db)
    oracle.apply_post(idempotency_key="k", body=_BODY)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE class_b_effects SET response_json = ? WHERE idempotency_key = ?",
            ("not-json", "k"),
        )
        conn.commit()
    fresh = ClassBHttpProviderOracle(db_path=db)
    with pytest.raises(ClassBHttpProviderOraclePersistenceError):
        fresh.apply_post(idempotency_key="k", body=_BODY)


def test_oracle_retention_enforced_from_sqlite_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "retention.db"
    oracle = ClassBHttpProviderOracle(db_path=db, max_retained_keys=2)
    oracle.apply_post(idempotency_key="a", body=b"a")
    oracle.apply_post(idempotency_key="b", body=b"b")
    restarted = ClassBHttpProviderOracle(db_path=db, max_retained_keys=2)
    with pytest.raises(ClassBHttpProviderOracleRetentionExceededError):
        restarted.apply_post(idempotency_key="c", body=b"c")
    replay = restarted.apply_post(idempotency_key="a", body=b"a")
    assert replay.replay


def test_oracle_http_contract_status_codes(tmp_path: Path) -> None:
    oracle = ClassBHttpProviderOracle(db_path=tmp_path / "http.db")
    handler = class_b_http_handler_for_oracle(oracle)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"

    def _post(key: str, body: bytes) -> int:
        req = urllib.request.Request(
            base,
            data=body,
            method="POST",
            headers={
                "Idempotency-Key": key,
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code

    assert _post("", _BODY) == 400
    assert _post("ok-1", _BODY) == 200
    assert _post("ok-1", _BODY) == 200
    assert _post("ok-2", _ALT_BODY) == 200
    assert _post("ok-2", _BODY) == 409
    server.shutdown()


def test_oracle_http_exactly_one_row_per_key(tmp_path: Path) -> None:
    db = tmp_path / "once.db"
    oracle = ClassBHttpProviderOracle(db_path=db)
    handler = class_b_http_handler_for_oracle(oracle)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    key = "parallel-http"

    def _post() -> None:
        req = urllib.request.Request(
            base,
            data=_BODY,
            method="POST",
            headers={
                "Idempotency-Key": key,
                "Content-Length": str(len(_BODY)),
            },
        )
        urllib.request.urlopen(req, timeout=10.0).read()

    workers = [threading.Thread(target=_post) for _ in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15.0)
    server.shutdown()
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM class_b_effects WHERE idempotency_key = ?",
            (key,),
        ).fetchone()[0]
    assert count == 1
    assert oracle.effect_count(key) == 1
