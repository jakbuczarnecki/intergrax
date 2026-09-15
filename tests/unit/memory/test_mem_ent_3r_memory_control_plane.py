# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-3R: typed capability boundary and lifecycle-aware control plane."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlForgetRequest,
    MemoryControlNotFound,
    MemoryControlPartialLifecycleError,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
    adapt_manager_search_result,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

pytestmark = pytest.mark.gate

_TENANT_A = "tenant-a"
_USER_U1 = "user-u1"


@dataclass
class RecordingMemoryProjection:
    projection_id: str = "mem_ent_3r_projection"
    upsert_calls: list[tuple[str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, ...]] = field(default_factory=list)
    fail_upsert: bool = False
    fail_delete: bool = False
    indexed_entry_ids: set[str] = field(default_factory=set)

    async def upsert_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if self.fail_upsert:
            raise TimeoutError("projection unavailable")
        self.upsert_calls.append((user_id, entry.entry_id))
        self.indexed_entry_ids.add(entry.entry_id)

    async def delete_memory_entries(self, entry_ids: Sequence[str]) -> None:
        if self.fail_delete:
            raise TimeoutError("projection unavailable")
        self.delete_calls.append(tuple(entry_ids))
        for entry_id in entry_ids:
            self.indexed_entry_ids.discard(entry_id)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        expected = set(context.authoritative_active_entry_ids)
        orphans = self.indexed_entry_ids - expected
        if orphans:
            await self.delete_memory_entries(tuple(sorted(orphans)))
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if orphans
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )


def _identity(user_id: str = _USER_U1) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT_A,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _plane(projection: RecordingMemoryProjection) -> DefaultMemoryControlPlane:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT_A,
        memory_projections=(projection,),
    )
    return DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )


@pytest.mark.asyncio
async def test_partial_remember_raises_typed_partial_lifecycle_error() -> None:
    projection = RecordingMemoryProjection(fail_upsert=True)
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="partial write"),
        )

    lifecycle = exc_info.value.lifecycle
    assert lifecycle.primary_applied is True
    assert lifecycle.requires_reconciliation is True
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE

    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    assert any(item.content == "partial write" for item in recall.items)


@pytest.mark.asyncio
async def test_partial_forget_raises_typed_partial_lifecycle_error() -> None:
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="to soft delete"),
    )
    assert remembered.entry_id in projection.indexed_entry_ids

    projection.fail_delete = True
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.forget(
            identity,
            scope,
            MemoryControlForgetRequest(entry_id=remembered.entry_id or ""),
        )

    lifecycle = exc_info.value.lifecycle
    assert lifecycle.primary_applied is True
    assert lifecycle.requires_reconciliation is True

    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    assert all(item.entry_id != remembered.entry_id for item in recall.items)


@pytest.mark.asyncio
async def test_total_backend_failure_before_primary_mutation() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection()
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT_A,
        memory_projections=(projection,),
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )

    async def fail_save(**kwargs: object) -> None:
        raise OSError("primary down")

    store.save_profile = fail_save  # type: ignore[method-assign]

    identity = _identity()
    scope = user_memory_scope(identity)
    with pytest.raises(MemoryControlBackendError):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="never saved"),
        )
    assert projection.upsert_calls == []


@pytest.mark.asyncio
async def test_remember_success_includes_consistent_lifecycle() -> None:
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    result = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="consistent"),
    )

    assert result.lifecycle is not None
    assert result.lifecycle.disposition is MemoryLifecycleDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_forget_not_found_maps_to_control_not_found() -> None:
    plane = _plane(RecordingMemoryProjection())
    identity = _identity()
    scope = user_memory_scope(identity)

    with pytest.raises(MemoryControlNotFound):
        await plane.forget(
            identity,
            scope,
            MemoryControlForgetRequest(entry_id="missing-entry"),
        )


def test_adapt_manager_search_result_typed_fields() -> None:
    entry = UserProfileMemoryEntry(entry_id="e1", content="fact")
    raw: dict[str, Any] = {
        "hits": [entry],
        "scores": [0.91],
        "debug": {"used": True, "reason": "vector"},
    }
    adapted = adapt_manager_search_result(raw)
    assert len(adapted.entries) == 1
    assert adapted.entries[0].entry_id == "e1"
    assert adapted.scores == (0.91,)
    assert adapted.used_semantic is True
    assert adapted.reason == "vector"


def test_adapt_manager_search_result_rejects_malformed_shape() -> None:
    with pytest.raises(MemoryControlBackendError):
        adapt_manager_search_result({"hits": "bad", "scores": []})


@dataclass
class _BrokenSearchManager:
    async def search_longterm_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        return {"hits": [1], "scores": []}


@pytest.mark.asyncio
async def test_capability_adapter_rejects_malformed_manager_search() -> None:
    capability = UserProfileManagerMemoryCapability(_manager=_BrokenSearchManager())  # type: ignore[arg-type]
    with pytest.raises(MemoryControlBackendError):
        await capability.search_longterm_memory(_USER_U1, "q")
