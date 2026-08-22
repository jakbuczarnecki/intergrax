# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Redis-backed IdempotencyStore implementation.

Composition root: ``intergrax.integrations.providers.key_value_cache.redis.create_redis_idempotency_store``.
"""

from __future__ import annotations

import base64
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.tools.execution_models import ToolExecutionResult


class RedisIdempotencyStore(IdempotencyStore):
    """
    Redis-backed implementation of IdempotencyStore.

    Ledger semantics: NONE -> STARTED (claim/lease/fence) -> COMPLETED.
    Expired claims without completion become UNCERTAIN.
    Shared authoritative state for multi-host workers when Redis is the backend.
    """

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.SHARED_MULTI_HOST

    def __init__(self, redis_client: Any) -> None:
        self._redis: Any = redis_client

        self._claim_script = self._redis.register_script(
            """
            -- KEYS[1] = ledger key
            -- ARGV[1] = owner_id
            -- ARGV[2] = lease_expires_at (ISO)
            -- ARGV[3] = now (ISO)
            --
            -- Returns array: {outcome_code, owner_id, lease_expires_at, fence, result_blob}
            -- outcome_code: acquired=1, replay=2, blocked=3, uncertain=4

            local status = redis.call("HGET", KEYS[1], "status")
            if not status then
                redis.call("HSET", KEYS[1],
                    "status", "started",
                    "owner_id", ARGV[1],
                    "lease_expires_at", ARGV[2],
                    "fence", "1")
                return {"1", ARGV[1], ARGV[2], "1", ""}
            end

            if status == "completed" then
                local blob = redis.call("HGET", KEYS[1], "result_blob") or ""
                return {"2", "", "", "", blob}
            end

            if status == "uncertain" then
                return {"4", "", "", "", ""}
            end

            local owner_id = redis.call("HGET", KEYS[1], "owner_id")
            local lease_expires_at = redis.call("HGET", KEYS[1], "lease_expires_at")
            local fence = redis.call("HGET", KEYS[1], "fence")

            if lease_expires_at > ARGV[3] then
                if owner_id == ARGV[1] then
                    return {"1", owner_id, lease_expires_at, fence, ""}
                end
                return {"3", owner_id, lease_expires_at, fence, ""}
            end

            redis.call("HSET", KEYS[1], "status", "uncertain")
            return {"4", owner_id, lease_expires_at, fence, ""}
            """
        )

        self._complete_with_claim_script = self._redis.register_script(
            """
            -- KEYS[1] = ledger key
            -- ARGV[1] = owner_id
            -- ARGV[2] = fence
            -- ARGV[3] = result_blob
            -- ARGV[4] = completed_ttl_seconds (optional)
            --
            -- Returns: 1 success, 0 missing, 2 stale/invalid

            if redis.call("EXISTS", KEYS[1]) == 0 then
                return 0
            end

            local status = redis.call("HGET", KEYS[1], "status")
            if status ~= "started" then
                return 2
            end

            local owner_id = redis.call("HGET", KEYS[1], "owner_id")
            local fence = redis.call("HGET", KEYS[1], "fence")
            if owner_id ~= ARGV[1] or fence ~= ARGV[2] then
                return 2
            end

            redis.call("HSET", KEYS[1], "status", "completed", "result_blob", ARGV[3])
            if ARGV[4] ~= "" then
                redis.call("EXPIRE", KEYS[1], tonumber(ARGV[4]))
            end
            return 1
            """
        )

        self._record_started_script = self._redis.register_script(
            """
            if redis.call("EXISTS", KEYS[1]) == 1 then
                return 0
            end
            redis.call("HSET", KEYS[1], "status", "started", "fence", "1")
            if ARGV[1] ~= "" then
                redis.call("EXPIRE", KEYS[1], tonumber(ARGV[1]))
            end
            return 1
            """
        )

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

    def _decode(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        ledger_key = self._ledger_key(tenant_id, key)
        raw = self._claim_script(
            keys=[ledger_key],
            args=[owner_id, lease_expires_at.isoformat(), now.isoformat()],
        )
        outcome_code = self._decode(raw[0])
        if outcome_code == "1":
            claim = InvocationClaim(
                tenant_id=tenant_id,
                key=key,
                owner_id=self._decode(raw[1]),
                lease_expires_at=datetime.fromisoformat(self._decode(raw[2])),
                fence=int(self._decode(raw[3])),
            )
            return ClaimResult(outcome=ClaimOutcome.ACQUIRED, claim=claim)
        if outcome_code == "2":
            blob = self._decode(raw[4])
            if not blob:
                raise RuntimeError("Ledger corruption: COMPLETED without result_blob")
            return ClaimResult(
                outcome=ClaimOutcome.REPLAY_COMPLETED,
                completed_result=self._deserialize_result(blob),
            )
        if outcome_code == "3":
            return ClaimResult(outcome=ClaimOutcome.BLOCKED_ACTIVE)
        return ClaimResult(outcome=ClaimOutcome.UNCERTAIN)

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        ledger_key = self._ledger_key(tenant_id, key)
        serialized = self._serialize_result(result)
        ttl_arg = str(completed_ttl_seconds) if completed_ttl_seconds is not None else ""
        script_result = self._complete_with_claim_script(
            keys=[ledger_key],
            args=[claim.owner_id, str(claim.fence), serialized, ttl_arg],
        )
        if script_result != 1:
            raise StaleClaimError(
                f"Stale completion rejected for key={key} fence={claim.fence}.",
            )

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        lease = lease_seconds if lease_seconds is not None else 300
        owner_id = f"legacy-{uuid4().hex}"
        outcome = self.claim(tenant_id, key, owner_id, lease)
        if outcome.outcome != ClaimOutcome.ACQUIRED:
            raise RuntimeError("Invocation already exists")

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        ledger_key = self._ledger_key(tenant_id, key)
        data = self._redis.hgetall(ledger_key)
        if not data:
            raise RuntimeError("Invalid state transition")
        owner_id = self._decode(data.get(b"owner_id", b""))
        lease_raw = data.get(b"lease_expires_at")
        fence_raw = data.get(b"fence", b"1")
        if not owner_id or lease_raw is None:
            raise RuntimeError("Invalid state transition")
        claim = InvocationClaim(
            tenant_id=tenant_id,
            key=key,
            owner_id=owner_id,
            lease_expires_at=datetime.fromisoformat(self._decode(lease_raw)),
            fence=int(self._decode(fence_raw)),
        )
        self.complete_with_claim(
            tenant_id,
            key,
            claim,
            result,
            completed_ttl_seconds=completed_ttl_seconds,
        )

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        ledger_key = self._ledger_key(tenant_id, key)
        status = self._redis.hget(ledger_key, "status")
        if status is None:
            return None
        status_value = self._decode(status)
        try:
            return InvocationStatus(status_value)
        except ValueError:
            raise RuntimeError(f"Invalid ledger status value: {status_value}")

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        ledger_key = self._ledger_key(tenant_id, key)
        data = self._redis.hgetall(ledger_key)
        if not data:
            return None
        status = data.get(b"status")
        if status is None:
            return None
        if self._decode(status) != InvocationStatus.COMPLETED.value:
            return None
        blob = data.get(b"result_blob")
        if blob is None:
            raise RuntimeError("Ledger corruption: COMPLETED without result_blob")
        return self._deserialize_result(self._decode(blob))
