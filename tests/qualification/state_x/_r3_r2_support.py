# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R2 shared qualification helpers (not collected by pytest directly)."""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pytest
from pydantic import BaseModel

from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
from intergrax.agents.persistence.compensation_queue_store import (
    CompensationClaim,
    CompensationJob,
    CompensationJobStatus,
    CompensationQueueStore,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.agents.persistence.compensation_side_effect_input import (
    compensation_side_effect_input_from_job,
)
from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
    execute_declarative_actions,
)
from intergrax.contracts.compensation_side_effect_execution import CompensationSideEffectInput
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    IdempotencyOperationConflictError,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
    PreEffectSuspendedWorkRecoveryAuthority,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.side_effect import CompensationRequest
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from tests.qualification.state_x.inventory import STATE_X_FAMILY_INVENTORY
from tests.unit.agents.persistence.compensation_execution_test_support import (
    RecordingExecutionBoundDeclarativeToolInvoker,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WIRING_PATH = _REPO_ROOT / "intergrax/applications/_shared/compensation_side_effect_wiring.py"
_TENANT_A = "tenant-r3r2-a"
_TENANT_B = "tenant-r3r2-b"


class _IdempotencyStoreFactory(Protocol):
    def __call__(self) -> IdempotencyStore: ...


def _inventory_by_id() -> dict[str, object]:
    return {entry.family_id: entry for entry in STATE_X_FAMILY_INVENTORY}


def _sqlite_idempotency_store() -> SQLiteIdempotencyStore:
    handle = tempfile.NamedTemporaryFile(suffix="-r3r2-idem.db", delete=False)
    handle.close()
    return SQLiteIdempotencyStore(handle.name)


def _local_idempotency_factories() -> tuple[_IdempotencyStoreFactory, ...]:
    return (
        InMemoryIdempotencyStore,
        _sqlite_idempotency_store,
    )


def _compensation_store_factories() -> tuple[Callable[[], CompensationQueueStore], ...]:
    def _sqlite(tmp_path: Path) -> Callable[[], CompensationQueueStore]:
        db = tmp_path / "comp-r3r2.db"

        def _factory() -> CompensationQueueStore:
            return SQLiteCompensationQueueStore(db)

        return _factory

    return (InMemoryCompensationQueueStore,)


class _DummyOut(BaseModel):
    value: int = 1


@dataclass
class _IdempotentBoundInvoker:
    """Execution-bound invoker with durable idempotency claim protocol."""

    _store: IdempotencyStore
    _inner: Callable[..., object]
    physical_calls: int = 0

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: dict[str, object],
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        if not idempotency_key:
            self.physical_calls += 1
            result = await self._inner(
                tenant_id=tenant_id,
                run_id=run_id,
                task_id=task_id,
                agent_id=agent_id,
                tool_id=tool_id,
                args=args,
                idempotency_key=idempotency_key,
            )
            return result
        claim_result = self._store.claim(
            tenant_id,
            idempotency_key,
            owner_id=f"{agent_id}:{run_id}",
            lease_seconds=30,
        )
        if claim_result.outcome == ClaimOutcome.REPLAY_COMPLETED:
            return DeclarativeToolInvokeResult(status="success")
        if claim_result.outcome == ClaimOutcome.BLOCKED_ACTIVE:
            return DeclarativeToolInvokeResult(status="denied", error="blocked_active")
        if claim_result.outcome == ClaimOutcome.UNCERTAIN:
            return DeclarativeToolInvokeResult(status="denied", error="uncertain")
        assert claim_result.claim is not None
        self.physical_calls += 1
        inner = await self._inner(
            tenant_id=tenant_id,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            tool_id=tool_id,
            args=args,
            idempotency_key=idempotency_key,
        )
        if inner.status == "success":
            result = ToolExecutionResult.ok(_DummyOut())
            self._store.complete_with_claim(
                tenant_id,
                idempotency_key,
                claim_result.claim,
                result,
            )
        return inner


def _sample_compensation_job(
    *,
    tenant_id: str = _TENANT_A,
    key_suffix: str = "job",
) -> CompensationJob:
    key = build_compensation_idempotency_key(f"acp:{key_suffix}")
    return CompensationJob(
        run_id=str(mint_run_id()),
        task_id=str(mint_task_id()),
        tenant_id=tenant_id,
        agent_id="agent-r3r2",
        step_index=0,
        request=CompensationRequest(
            original_side_effect_id="se-r3r2",
            compensation_tool_id="email.recall",
            args={"ref": "x"},
            idempotency_key=key,
        ),
    )


def _build_admitted_compensation_execution(
    invoker: ExecutionBoundDeclarativeToolInvoker,
) -> object:
    return build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(invoker),
        authority=ParentExecutionAuthority.unrestricted_root(),
    )


def _redis_store_or_skip() -> object:
    pytest.importorskip("redis")
    from redis import Redis

    from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore

    client = Redis(
        host="localhost",
        port=6379,
        decode_responses=False,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis unavailable for R3-R2 qualification: {exc}")
    return RedisIdempotencyStore(redis_client=client)


