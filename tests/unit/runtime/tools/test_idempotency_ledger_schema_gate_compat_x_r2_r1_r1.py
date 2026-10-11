# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R1-R1 adversarial schema gates (SQLite; Redis when available)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
    ClaimOutcome,
    IdempotencyLedgerSchemaError,
    InvocationClaim,
    InvocationOperationIdentity,
    InvocationStatus,
    PreEffectSuspendedWorkRecoveryAuthority,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.unit


class _Out(BaseModel):
    value: int = 1


def _result() -> ToolExecutionResult[_Out]:
    return ToolExecutionResult(success=True, output=_Out(), error=None)


def _valid_claim(tenant: str, key: str) -> InvocationClaim:
    return InvocationClaim(
        tenant_id=tenant,
        key=key,
        owner_id="owner-a",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        fence=1,
    )


def _operation_identity() -> InvocationOperationIdentity:
    return InvocationOperationIdentity(
        tool_id="tool",
        operation_fingerprint="sha256:" + "a" * 64,
    )


def _insert_started_row(
    store: SQLiteIdempotencyStore,
    *,
    tenant: str,
    key: str,
    schema_version: str | None,
) -> None:
    lease = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO idempotency_ledger
                (tenant_id, key, status, owner_id, lease_expires_at, fence,
                 external_effect_may_have_started, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                tenant,
                key,
                InvocationStatus.STARTED.value,
                "owner-a",
                lease,
                1,
                schema_version,
            ),
        )


def _row_snapshot(
    store: SQLiteIdempotencyStore, tenant: str, key: str
) -> sqlite3.Row | None:
    with store._connect() as conn:
        return conn.execute(
            """
            SELECT status, result_blob, external_effect_may_have_started,
                   schema_version, fence
            FROM idempotency_ledger
            WHERE tenant_id = ? AND key = ?
            """,
            (tenant, key),
        ).fetchone()


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteIdempotencyStore:
    return SQLiteIdempotencyStore(str(tmp_path / "idempotency-schema-gate.db"))


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_sqlite_complete_with_claim_rejects_legacy_or_unknown_schema(
    sqlite_store: SQLiteIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "t1", "complete-gate"
    _insert_started_row(
        sqlite_store, tenant=tenant, key=key, schema_version=schema_version
    )
    before = _row_snapshot(sqlite_store, tenant, key)
    claim = _valid_claim(tenant, key)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.complete_with_claim(tenant, key, claim, _result())
    after = _row_snapshot(sqlite_store, tenant, key)
    assert dict(before) == dict(after)


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_sqlite_mark_uncertain_rejects_legacy_or_unknown_schema(
    sqlite_store: SQLiteIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "t1", "uncertain-gate"
    _insert_started_row(
        sqlite_store, tenant=tenant, key=key, schema_version=schema_version
    )
    before = _row_snapshot(sqlite_store, tenant, key)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.mark_uncertain_with_claim(tenant, key, _valid_claim(tenant, key))
    assert dict(before) == dict(_row_snapshot(sqlite_store, tenant, key))


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_sqlite_abandon_rejects_legacy_or_unknown_schema(
    sqlite_store: SQLiteIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "t1", "abandon-gate"
    _insert_started_row(
        sqlite_store, tenant=tenant, key=key, schema_version=schema_version
    )
    before = _row_snapshot(sqlite_store, tenant, key)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.abandon_pre_effect_with_claim(
            tenant, key, _valid_claim(tenant, key)
        )
    assert dict(before) == dict(_row_snapshot(sqlite_store, tenant, key))


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_sqlite_admit_external_effect_rejects_legacy_or_unknown_schema(
    sqlite_store: SQLiteIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "t1", "admit-gate"
    _insert_started_row(
        sqlite_store, tenant=tenant, key=key, schema_version=schema_version
    )
    before = _row_snapshot(sqlite_store, tenant, key)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.admit_external_effect_may_have_started_with_claim(
            tenant, key, _valid_claim(tenant, key)
        )
    assert dict(before) == dict(_row_snapshot(sqlite_store, tenant, key))


def test_sqlite_external_effect_read_fail_closed_without_schema(
    sqlite_store: SQLiteIdempotencyStore,
) -> None:
    tenant, key = "t1", "effect-read"
    _insert_started_row(sqlite_store, tenant=tenant, key=key, schema_version=None)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.external_effect_may_have_started(tenant, key)


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_sqlite_reconcile_rejects_legacy_or_unknown_schema(
    sqlite_store: SQLiteIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "t1", "reconcile-gate"
    _insert_started_row(
        sqlite_store, tenant=tenant, key=key, schema_version=schema_version
    )
    before = _row_snapshot(sqlite_store, tenant, key)
    authority = PreEffectSuspendedWorkRecoveryAuthority(owner_id="host", fence=2)
    with pytest.raises(IdempotencyLedgerSchemaError):
        sqlite_store.reconcile_abandoned_pre_effect_not_started(
            tenant,
            key,
            _operation_identity(),
            recovery_authority=authority,
        )
    assert dict(before) == dict(_row_snapshot(sqlite_store, tenant, key))


def test_sqlite_v1_mutation_paths_remain_valid(sqlite_store: SQLiteIdempotencyStore) -> None:
    tenant, key = "t1", "happy-path"
    acquired = sqlite_store.claim(tenant, key, "owner-a", 300, _operation_identity())
    assert acquired.outcome is ClaimOutcome.ACQUIRED
    claim = acquired.claim
    assert claim is not None
    sqlite_store.admit_external_effect_may_have_started_with_claim(tenant, key, claim)
    assert sqlite_store.external_effect_may_have_started(tenant, key)
    sqlite_store.complete_with_claim(tenant, key, claim, _result())
    assert sqlite_store.get_status(tenant, key) is InvocationStatus.COMPLETED
    replay = sqlite_store.get_completed_result(tenant, key)
    assert replay is not None and replay.success


def _redis_available() -> bool:
    try:
        from redis import Redis

        client = Redis(host="localhost", port=6379, decode_responses=False)
        return bool(client.ping())
    except Exception:
        return False


@pytest.fixture
def redis_store() -> RedisIdempotencyStore:
    if not _redis_available():
        pytest.skip("Redis not available on localhost:6379")
    from redis import Redis

    return RedisIdempotencyStore(Redis(host="localhost", port=6379, decode_responses=False))


def _redis_seed_started(
    store: RedisIdempotencyStore,
    tenant: str,
    key: str,
    *,
    schema_version: str | None,
) -> None:
    ledger_key = store._ledger_key(tenant, key)
    lease = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    mapping: dict[str, str] = {
        "status": InvocationStatus.STARTED.value,
        "owner_id": "owner-a",
        "lease_expires_at": lease,
        "fence": "1",
        "external_effect_may_have_started": "0",
        "operation_tool_id": "tool",
        "operation_fingerprint": "sha256:" + "b" * 64,
    }
    if schema_version is not None:
        mapping["schema_version"] = schema_version
    store._redis.hset(ledger_key, mapping=mapping)


def _redis_hash_snapshot(store: RedisIdempotencyStore, tenant: str, key: str) -> dict:
    return store._redis.hgetall(store._ledger_key(tenant, key))


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_redis_complete_rejects_legacy_or_unknown_schema(
    redis_store: RedisIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "rt", f"redis-complete-{schema_version}"
    _redis_seed_started(redis_store, tenant, key, schema_version=schema_version)
    before = _redis_hash_snapshot(redis_store, tenant, key)
    with pytest.raises(IdempotencyLedgerSchemaError):
        redis_store.complete_with_claim(
            tenant, key, _valid_claim(tenant, key), _result()
        )
    assert _redis_hash_snapshot(redis_store, tenant, key) == before


@pytest.mark.parametrize("schema_version", [None, "tool_idempotency_ledger.v999"])
def test_redis_external_effect_read_fail_closed(
    redis_store: RedisIdempotencyStore,
    schema_version: str | None,
) -> None:
    tenant, key = "rt", f"redis-effect-read-{schema_version}"
    _redis_seed_started(redis_store, tenant, key, schema_version=schema_version)
    with pytest.raises(IdempotencyLedgerSchemaError):
        redis_store.external_effect_may_have_started(tenant, key)


def test_sqlite_stale_claim_still_raises_stale_not_schema(
    sqlite_store: SQLiteIdempotencyStore,
) -> None:
    tenant, key = "t1", "stale-distinction"
    _insert_started_row(
        sqlite_store,
        tenant=tenant,
        key=key,
        schema_version=SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
    )
    bad_claim = InvocationClaim(
        tenant_id=tenant,
        key=key,
        owner_id="other-owner",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        fence=1,
    )
    with pytest.raises(StaleClaimError):
        sqlite_store.complete_with_claim(tenant, key, bad_claim, _result())
