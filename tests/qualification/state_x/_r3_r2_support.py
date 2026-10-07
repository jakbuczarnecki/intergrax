# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R2 shared qualification helpers (not collected by pytest directly)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest
from pydantic import BaseModel

from intergrax.agents.persistence.compensation_enqueue import build_compensation_idempotency_key
from intergrax.agents.persistence.compensation_queue_store import (
    CompensationJob,
    CompensationQueueStore,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.applications._shared.compensation_side_effect_wiring import (
    build_compensation_side_effect_execution,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
    CompensationSideEffectInput,
    CompensationSideEffectInvokeResult,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.side_effect import CompensationRequest
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.qualification.state_x.inventory import STATE_X_FAMILY_INVENTORY

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WIRING_PATH = _REPO_ROOT / "intergrax/applications/_shared/compensation_side_effect_wiring.py"
_TENANT_A = "tenant-r3r2-a"
_TENANT_B = "tenant-r3r2-b"

_COMPENSATION_TOOL_ID = "email.recall"


class _IdempotencyStoreFactory(Protocol):
    def __call__(self) -> IdempotencyStore: ...


class CompensationQueueStoreFactory(Protocol):
    def __call__(self, tmp_path: Path) -> CompensationQueueStore: ...


class IdempotencyStorePathFactory(Protocol):
    def __call__(self, tmp_path: Path) -> IdempotencyStore: ...


class _RecallIn(BaseModel):
    ref: str = "x"


class _RecallOut(BaseModel):
    ok: bool = True


class _CountingRecallHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest) -> _RecallOut:
        self.calls += 1
        return _RecallOut()


@dataclass
class _LabPolicyCatalogDeclarativeToolInvoker(CatalogDeclarativeToolInvoker):
    """Catalog invoker with lab ENFORCE policy bundle on dispatch state (qualification only)."""

    def _runtime_state(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        user_id: str,
    ) -> RuntimeState:
        state = super()._runtime_state(
            tenant_id=tenant_id,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            user_id=user_id,
        )
        env = ApplicationEnvironmentProfile.lab_defaults(profile_id="qual.r3r2.comp")
        env.policy_rules = PolicyRulesProfile(
            inline_rules=[],
            policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
        )
        state.context.config.policy_bundle = wire_policy_bundle(env)
        return state


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


def _in_memory_compensation_queue_store(tmp_path: Path) -> CompensationQueueStore:
    return InMemoryCompensationQueueStore()


def _sqlite_compensation_queue_store(tmp_path: Path) -> CompensationQueueStore:
    return SQLiteCompensationQueueStore(tmp_path / "compensation-r3r2.db")


def _compensation_queue_store_factories() -> tuple[CompensationQueueStoreFactory, ...]:
    return (_in_memory_compensation_queue_store, _sqlite_compensation_queue_store)


def _in_memory_idempotency_store_path(tmp_path: Path) -> IdempotencyStore:
    return InMemoryIdempotencyStore()


def _sqlite_idempotency_store_path(tmp_path: Path) -> IdempotencyStore:
    return SQLiteIdempotencyStore(str(tmp_path / "idempotency-r3r2.db"))


def _durable_idempotency_store_factories() -> tuple[IdempotencyStorePathFactory, ...]:
    return (_in_memory_idempotency_store_path, _sqlite_idempotency_store_path)


def _paired_compensation_idempotency_factories() -> tuple[
    tuple[CompensationQueueStoreFactory, IdempotencyStorePathFactory],
    ...,
]:
    return (
        (_in_memory_compensation_queue_store, _in_memory_idempotency_store_path),
        (_sqlite_compensation_queue_store, _sqlite_idempotency_store_path),
    )


async def _execute_compensation_with_lab_governance(
    execution: CompensationSideEffectExecutionPort,
    work: CompensationSideEffectInput,
) -> CompensationSideEffectInvokeResult:
    """Bind lab governance identity required by declarative policy evidence on tool invoke."""
    from intergrax.runtime.governance.active_execution_governance_identity import (
        ActiveExecutionGovernanceIdentity,
        bind_active_execution_governance_identity,
        reset_active_execution_governance_identity,
    )

    token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=work.tenant_id,
            workspace_id="ws-r3r2-qualification",
            principal_id="principal-r3r2-qualification",
        ),
    )
    try:
        return await execution.execute(work)
    finally:
        reset_active_execution_governance_identity(token)


def _build_admitted_compensation_execution(
    invoker: ExecutionBoundDeclarativeToolInvoker,
) -> CompensationSideEffectExecutionPort:
    return build_compensation_side_effect_execution(
        invoker,
        authority=ParentExecutionAuthority.unrestricted_root(),
    )


def _build_canonical_compensation_production_execution(
    idempotency_store: IdempotencyStore,
) -> tuple[
    CompensationSideEffectExecutionPort,
    _CountingRecallHandler,
    CatalogDeclarativeToolInvoker,
]:
    """Materialize compensation admission through catalog + RuntimeToolInvoker idempotency."""
    registry = ToolRegistry()
    handler = _CountingRecallHandler()
    registry.register(
        contract=ToolContract(
            tool_id=_COMPENSATION_TOOL_ID,
            name=_COMPENSATION_TOOL_ID,
            description="qualification recall side effect",
            input_schema=_RecallIn,
            output_schema=_RecallOut,
            error_mapping={},
            side_effects=True,
        ),
        handler=handler,
    )
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=[_COMPENSATION_TOOL_ID]),
        wiring_context=ToolWiringContext(),
        registry=registry,
    )
    catalog = build_declarative_invoker_from_tool_wiring(
        wiring,
        idempotency_store=idempotency_store,
        production_mode=False,
    )
    if catalog is None:
        raise RuntimeError("catalog declarative invoker required for R3-R2 production-path proof")
    catalog_with_policy = _LabPolicyCatalogDeclarativeToolInvoker(
        tool_invoker=catalog.tool_invoker,
        binding=catalog.binding,
        production_mode=catalog.production_mode,
    )
    port = build_compensation_side_effect_execution(
        catalog_with_policy,
        authority=ParentExecutionAuthority.unrestricted_root(),
    )
    return port, handler, catalog_with_policy


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
            compensation_tool_id=_COMPENSATION_TOOL_ID,
            args={"ref": "x"},
            idempotency_key=key,
        ),
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

