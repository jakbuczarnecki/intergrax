# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import pickle
from typing import Optional

from pydantic import BaseModel
from redis import Redis

from intergrax.contracts.idempotency_store import (
    IdempotencyStore,
    InvocationStatus,
)
from intergrax.tools.execution_models import ToolExecutionResult


class RedisIdempotencyStore(IdempotencyStore):
    """
    Redis-backed implementation of IdempotencyStore.

    Ledger semantics:

        NONE -> STARTED (lease) -> COMPLETED

    Guarantees:
    - atomic STARTED insertion
    - atomic STARTED -> COMPLETED transition
    - deterministic result replay
    - multi-tenant isolation
    - optional lease-based crash recovery
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis: Redis = redis_client

        # Lua script: record_started with optional lease
        self._record_started_script = self._redis.register_script(
            """
            -- KEYS[1] = ledger key
            -- ARGV[1] = lease_seconds (optional, empty string if None)
            --
            -- Returns:
            --   1  -> success
            --   0  -> already exists

            if redis.call("EXISTS", KEYS[1]) == 1 then
                return 0
            end

            redis.call("HSET", KEYS[1], "status", "started")

            if ARGV[1] ~= "" then
                redis.call("EXPIRE", KEYS[1], tonumber(ARGV[1]))
            end

            return 1
            """
        )

        # Lua script: record_completed with optional TTL
        self._record_completed_script = self._redis.register_script(
            """
            -- KEYS[1] = ledger key
            -- ARGV[1] = result_blob
            -- ARGV[2] = completed_ttl_seconds (optional, empty string if None)
            --
            -- Returns:
            --   1 -> success
            --   0 -> key does not exist
            --   2 -> invalid state

            if redis.call("EXISTS", KEYS[1]) == 0 then
                return 0
            end

            local status = redis.call("HGET", KEYS[1], "status")

            if status ~= "started" then
                return 2
            end

            redis.call("HSET", KEYS[1], "status", "completed")
            redis.call("HSET", KEYS[1], "result_blob", ARGV[1])

            if ARGV[2] ~= "" then
                redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
            end

            return 1
            """
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _ledger_key(tenant_id: str, key: str) -> str:
        return f"idempotency:{tenant_id}:{key}"

    @staticmethod
    def _serialize_result(result: ToolExecutionResult[BaseModel]) -> str:
        blob: bytes = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(blob).decode("utf-8")

    @staticmethod
    def _deserialize_result(blob: str) -> ToolExecutionResult[BaseModel]:
        raw: bytes = base64.b64decode(blob.encode("utf-8"))
        return pickle.loads(raw)

    # ---------------------------------------------------------------------
    # Contract implementation
    # ---------------------------------------------------------------------

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        ledger_key: str = self._ledger_key(tenant_id, key)

        lease_arg: str = str(lease_seconds) if lease_seconds is not None else ""

        result = self._record_started_script(
            keys=[ledger_key],
            args=[lease_arg],
        )

        if result != 1:
            raise RuntimeError("Invocation already exists")

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        ledger_key: str = self._ledger_key(tenant_id, key)
        serialized: str = self._serialize_result(result)

        ttl_arg: str = (
            str(completed_ttl_seconds)
            if completed_ttl_seconds is not None
            else ""
        )

        script_result = self._record_completed_script(
            keys=[ledger_key],
            args=[serialized, ttl_arg],
        )

        if script_result != 1:
            raise RuntimeError("Invalid state transition")

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        ledger_key: str = self._ledger_key(tenant_id, key)

        status = self._redis.hget(ledger_key, "status")
        if status is None:
            return None

        if isinstance(status, bytes):
            status_value = status.decode("utf-8")
        else:
            status_value = status

        try:
            return InvocationStatus(status_value)
        except ValueError:
            raise RuntimeError(f"Invalid ledger status value: {status_value}")

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        ledger_key: str = self._ledger_key(tenant_id, key)

        data = self._redis.hgetall(ledger_key)
        if not data:
            return None

        status = data.get(b"status")
        if status is None:
            return None

        status_value = status.decode("utf-8")
        if status_value != InvocationStatus.COMPLETED.value:
            return None

        blob = data.get(b"result_blob")
        if blob is None:
            raise RuntimeError(
                "Ledger corruption: COMPLETED without result_blob"
            )

        return self._deserialize_result(blob.decode("utf-8"))