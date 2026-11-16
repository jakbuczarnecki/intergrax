# © Artur Czarnecki. All rights reserved.

"""PDC-D2 R015: retention expiry enforced through MemoryControlPlane with vector tombstones."""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlNotFound,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationResult,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.memory_vector_namespace import LTM_INDEX_DOMAIN, resolve_memory_index_collection
from intergrax.memory.retention_sweep import (
    MEMORY_RETENTION_SWEEP_TASK_NAME,
    MemoryRetentionSweep,
    MemoryRetentionSweepRequest,
    memory_retention_sweep_task_request,
    register_memory_retention_sweep_task,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.rag.vectorstore.contracts.native_vectorstore import MetadataFilter, VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from tests.unit.background_tasks.test_worker_runtime import _KV
from tests.unit.runtime.background_execution.reentry_admission_doubles import (
    admission_kwargs,
    make_kv_admission_dependencies,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-d2"
_NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


class _FixedEmbedding:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _identity(user_id: str = "user-1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


@dataclass
class _Harness:
    plane: DefaultMemoryControlPlane
    capability: UserProfileManagerMemoryCapability
    vector_manager: VectorstoreManager
    projection: object
    remembered: dict[str, str] = field(default_factory=dict)

    def indexed_ids(self, user_id: str = "user-1") -> set[str]:
        ids = self.vector_manager.list_vector_ids_by_metadata(
            scope=VectorStoreScope(tenant_id=_TENANT),
            metadata_filter=MetadataFilter(
                conditions={
                    "user_id": user_id,
                    "index_domain": LTM_INDEX_DOMAIN,
                    "collection_name": resolve_memory_index_collection(
                        vector_index_namespace=None,
                        tenant_id=_TENANT,
                        domain=LTM_INDEX_DOMAIN,
                    ),
                }
            ),
            limit=1000,
        )
        return set(ids)

    async def remember(
        self,
        label: str,
        *,
        user_id: str = "user-1",
        valid_until: datetime | None = None,
    ) -> str:
        identity = _identity(user_id)
        entry = UserProfileMemoryEntry(
            content=f"memory-{label}",
            valid_from=(_NOW - timedelta(days=30)).isoformat(),
            valid_until=valid_until.isoformat() if valid_until is not None else None,
        )
        result = await self.plane.remember(
            identity,
            user_memory_scope(identity),
            MemoryControlRememberRequest(entry=entry),
        )
        assert result.entry_id
        self.remembered[label] = result.entry_id
        return result.entry_id


def _harness(projection_factory=None) -> _Harness:  # type: ignore[no-untyped-def]
    backend = InMemoryVectorStore(_TENANT)
    manager = VectorstoreManager(backend, scope=VectorStoreScope(tenant_id=_TENANT))
    real_projection = UserProfileLtmVectorProjection(
        embedding_manager=_FixedEmbedding(),  # type: ignore[arg-type]
        vectorstore_manager=manager,
        tenant_id=_TENANT,
        vector_index_namespace=None,
        workspace_id=None,
    )
    projection = projection_factory(real_projection) if projection_factory else real_projection
    profile_manager = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id=_TENANT,
        memory_projections=(projection,),
    )
    capability = UserProfileManagerMemoryCapability(_manager=profile_manager)
    plane = DefaultMemoryControlPlane(user_profile=capability)
    return _Harness(
        plane=plane,
        capability=capability,
        vector_manager=manager,
        projection=projection,
    )


async def _active_ids(harness: _Harness, user_id: str = "user-1") -> set[str]:
    return {e.entry_id for e in await harness.capability.list_active_memory_entries(user_id)}


@pytest.mark.asyncio
async def test_expired_memory_is_forgotten_and_its_vector_entry_tombstoned() -> None:
    harness = _harness()
    expired = await harness.remember("expired", valid_until=_NOW - timedelta(days=1))
    expired_two = await harness.remember("expired-two", valid_until=_NOW - timedelta(hours=1))
    future = await harness.remember("future", valid_until=_NOW + timedelta(days=30))
    forever = await harness.remember("forever")
    assert harness.indexed_ids() == {expired, expired_two, future, forever}

    report = await MemoryRetentionSweep(
        control_plane=harness.plane,
        candidates=harness.capability,
    ).run(MemoryRetentionSweepRequest(identity=_identity(), as_of=_NOW))

    assert set(report.forgotten_entry_ids) == {expired, expired_two}
    assert set(report.tombstoned_vector_entry_ids) == {expired, expired_two}
    assert report.failed_entry_ids == ()
    assert report.scanned == 4
    # Vector index: expired entries are gone, the others are untouched.
    assert harness.indexed_ids() == {future, forever}
    # Primary store: forgotten entries are soft-deleted, the others stay recallable.
    assert await _active_ids(harness) == {future, forever}
    profile = await harness.capability._manager.get_profile("user-1")
    assert {e.entry_id for e in profile.memory_entries if e.deleted} == {expired, expired_two}


@pytest.mark.asyncio
async def test_sweep_is_idempotent_and_scoped_to_the_requesting_identity() -> None:
    harness = _harness()
    mine = await harness.remember("mine", valid_until=_NOW - timedelta(days=1))
    theirs = await harness.remember("theirs", user_id="user-2", valid_until=_NOW - timedelta(days=1))

    sweep = MemoryRetentionSweep(control_plane=harness.plane, candidates=harness.capability)
    first = await sweep.run(MemoryRetentionSweepRequest(identity=_identity(), as_of=_NOW))
    second = await sweep.run(MemoryRetentionSweepRequest(identity=_identity(), as_of=_NOW))

    assert first.forgotten_entry_ids == (mine,)
    assert second.forgotten_entry_ids == ()
    assert second.failed_entry_ids == ()
    assert harness.indexed_ids("user-1") == set()
    assert harness.indexed_ids("user-2") == {theirs}


@pytest.mark.asyncio
async def test_expired_entry_is_not_forgettable_without_the_retention_path() -> None:
    harness = _harness()
    expired = await harness.remember("expired", valid_until=_NOW - timedelta(days=1))
    identity = _identity()

    with pytest.raises(MemoryControlNotFound):
        await harness.plane.forget(
            identity,
            user_memory_scope(identity),
            MemoryControlForgetRequest(entry_id=expired),
        )
    assert harness.indexed_ids() == {expired}


@dataclass
class _FlakyDeleteProjection:
    """Real LTM projection whose first delete fails (retryable partial lifecycle)."""

    inner: UserProfileLtmVectorProjection
    failures_left: int = 1

    @property
    def projection_id(self) -> str:
        return self.inner.projection_id

    async def upsert_memory_entry(self, context, entry) -> None:  # type: ignore[no-untyped-def]
        await self.inner.upsert_memory_entry(context, entry)

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("vector backend unavailable")
        await self.inner.delete_memory_entries(context, entry_ids)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        return await self.inner.reconcile(context)


@pytest.mark.asyncio
async def test_failed_vector_delete_is_reported_and_repaired_by_reconciliation() -> None:
    harness = _harness(lambda real: _FlakyDeleteProjection(real))
    expired = await harness.remember("expired", valid_until=_NOW - timedelta(days=1))
    kept = await harness.remember("kept")

    report = await MemoryRetentionSweep(
        control_plane=harness.plane,
        candidates=harness.capability,
    ).run(MemoryRetentionSweepRequest(identity=_identity(), as_of=_NOW))

    assert report.partial_entry_ids == (expired,)
    assert report.reconciled is True
    # Primary delete applied, projection delete failed, reconcile removed the orphan vector.
    assert harness.indexed_ids() == {kept}
    assert await _active_ids(harness) == {kept}


# ----------------------------------------------------------------- background admission


def _runtime(task_registry: TaskRegistry) -> WorkerRuntime:
    kv = _KV()
    execution_registry = TaskExecutionRegistry()
    task_registry.bind_execution_registry(execution_registry)
    return WorkerRuntime(
        registry=task_registry,
        state_store=TaskStateStore(kv_store=kv, provider="kafka"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kafka",
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        **admission_kwargs(make_kv_admission_dependencies(kv)),  # pyright: ignore[reportArgumentType]
    )


def test_retention_sweep_is_admitted_as_background_work_through_the_task_runtime() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))
    kept = asyncio.run(harness.remember("kept"))

    registry = TaskRegistry()
    register_memory_retention_sweep_task(
        registry,
        sweep=MemoryRetentionSweep(control_plane=harness.plane, candidates=harness.capability),
        identity_resolver=lambda tenant_id, user_id: _identity(user_id),
        clock=lambda: _NOW,
    )
    assert registry.has_task(MEMORY_RETENTION_SWEEP_TASK_NAME)

    request = memory_retention_sweep_task_request(tenant_id=_TENANT, run_id="run-retention-1", user_id="user-1")
    result = _runtime(registry).process_request(request, task_id="run-retention-1")

    assert result.status == TaskStatus.SUCCEEDED
    assert harness.indexed_ids() == {kept}
    assert asyncio.run(_active_ids(harness)) == {kept}
    assert expired not in harness.indexed_ids()


def test_background_retention_sweep_rejects_a_payload_for_another_tenant() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))

    registry = TaskRegistry()
    register_memory_retention_sweep_task(
        registry,
        sweep=MemoryRetentionSweep(control_plane=harness.plane, candidates=harness.capability),
        identity_resolver=lambda tenant_id, user_id: _identity(user_id),
        clock=lambda: _NOW,
    )
    forged = memory_retention_sweep_task_request(tenant_id=_TENANT, run_id="run-retention-2", user_id="user-1")
    forged_payload = json.loads(forged.payload.decode("utf-8"))
    forged_payload["tenant_id"] = "tenant-other"
    forged = type(forged)(
        tenant_id=forged.tenant_id,
        run_id=forged.run_id,
        task_name=forged.task_name,
        payload=json.dumps(forged_payload).encode("utf-8"),
    )
    result = _runtime(registry).process_request(forged, task_id="run-retention-2")

    assert result.status != TaskStatus.SUCCEEDED
    assert harness.indexed_ids() == {expired}


def test_retention_sweep_adds_no_scheduler_of_its_own() -> None:
    source = Path(__file__).resolve().parents[4] / "intergrax" / "memory" / "retention_sweep.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"sched", "apscheduler", "schedule", "celery", "croniter"}
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"Timer", "call_later", "call_at", "create_task"}
    assert "TaskDefinition" in {
        alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for alias in n.names
    }
    assert TaskDefinition.__name__ == "TaskDefinition"
