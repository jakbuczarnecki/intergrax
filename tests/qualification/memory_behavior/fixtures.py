# © Artur Czarnecki. All rights reserved.

"""Shared fixtures and contract-based fakes for behavioral memory evals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.memory.contracts.memory_control import (
    MemoryControlPlane,
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlScopeRef,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
    user_profile_memory_projection_context,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemorySecurityStrategySet,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

TENANT_A = "mem-audit6-tenant-a"
TENANT_B = "mem-audit6-tenant-b"


class FailingDeleteLtmVectorProjection(UserProfileLtmVectorProjection):
    """Fault-injecting LTM projection leaving stale vectors after canonical delete."""

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        raise TimeoutError("projection delete failed — stale vector retained")


class FixedEmbeddingManager:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "polish" in lowered or "polski" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "postgresql" in lowered or "database" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "language" in lowered or "język" in lowered:
                vectors.append([0.9, 0.1, 0.0])
            else:
                vectors.append([0.1, 0.1, 0.8])
        return vectors


def request_identity(
    *,
    tenant_id: str = TENANT_A,
    user_id: str,
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def task_scope(
    identity: RequestIdentity,
    *,
    namespace: str,
    key: str,
) -> MemoryControlScopeRef:
    return MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.TASK,
        tenant_id=identity.tenant_id,
        task_namespace=namespace,
        task_key=key,
    )


def session_scope(
    identity: RequestIdentity,
    *,
    session_id: str,
) -> MemoryControlScopeRef:
    return MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.SESSION,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        session_id=session_id,
    )


@dataclass
class RecordingMemoryProjection:
    projection_id: str = "audit6_projection"
    upsert_calls: list[tuple[str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, ...]] = field(default_factory=list)
    indexed_entry_ids: set[str] = field(default_factory=set)
    fail_upsert: bool = False
    fail_delete: bool = False

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if self.fail_upsert:
            raise TimeoutError("projection upsert failed")
        self.upsert_calls.append((context.user_id, entry.entry_id))
        self.indexed_entry_ids.add(entry.entry_id)

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        if self.fail_delete:
            raise TimeoutError("projection delete failed")
        self.delete_calls.append(tuple(entry_ids))
        for entry_id in entry_ids:
            self.indexed_entry_ids.discard(entry_id)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        expected = set(context.authoritative_active_entry_ids)
        orphans = self.indexed_entry_ids - expected
        changed = bool(orphans)
        if orphans:
            await self.delete_memory_entries(
                user_profile_memory_projection_context(context.identity),
                tuple(sorted(orphans)),
            )
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if changed
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )


def deny_remember_governance() -> MemorySecurityGovernanceService:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="audit6.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    class DenyRememberAuthorization:
        policy_id = "audit6.deny"
        policy_version = "1"

        def evaluate(
            self, request: MemoryGovernanceEvaluationRequest
        ) -> MemoryGovernanceDecision:
            return denied

    strategies = build_default_memory_security_strategy_set()
    custom = MemorySecurityStrategySet(
        authorization=DenyRememberAuthorization(),
        trust=strategies.trust,
        admission=strategies.admission,
        governance=strategies.governance,
        retention=strategies.retention,
    )
    return MemorySecurityGovernanceService(strategies=custom)


def build_user_control_plane(
    *,
    tenant_id: str = TENANT_A,
    projection: RecordingMemoryProjection | UserProfileLtmVectorProjection | None = None,
    ltm_projection: UserProfileLtmVectorProjection | None = None,
    enable_ltm_vector: bool = False,
    governance: MemorySecurityGovernanceService | None = None,
    observability_sink: Any | None = None,
) -> tuple[MemoryControlPlane, UserProfileManager]:
    store = InMemoryUserProfileStore()
    projection_list: list[Any] = []
    manager_kwargs: dict[str, Any] = {"store": store, "tenant_id": tenant_id}
    if enable_ltm_vector:
        backend = InMemoryVectorStore(tenant_id)
        vector_manager = VectorstoreManager(backend, scope=VectorStoreScope(tenant_id=tenant_id))
        ltm_impl = ltm_projection or UserProfileLtmVectorProjection(
            embedding_manager=FixedEmbeddingManager(),  # type: ignore[arg-type]
            vectorstore_manager=vector_manager,
            tenant_id=tenant_id,
            vector_index_namespace=None,
            workspace_id=None,
        )
        projection_list.append(ltm_impl)
        manager_kwargs["embedding_manager"] = FixedEmbeddingManager()  # type: ignore[arg-type]
        manager_kwargs["vectorstore_manager"] = vector_manager
    if projection is not None:
        projection_list.append(projection)
    manager_kwargs["memory_projections"] = tuple(projection_list)
    user_manager = UserProfileManager(**manager_kwargs)
    plane = build_default_memory_control_plane(
        user_profile_manager=user_manager,
        security_governance=governance,
        memory_observability_sink=observability_sink,
    )
    return plane, user_manager


@dataclass
class InMemoryTaskMemoryCapability:
    _data: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)

    async def write(self, namespace: str, key: str, value: dict[str, object]) -> None:
        self._data[(namespace, key)] = dict(value)

    async def read(self, namespace: str, key: str) -> dict[str, object] | None:
        return self._data.get((namespace, key))

    async def delete(self, namespace: str, key: str) -> bool:
        return self._data.pop((namespace, key), None) is not None


@dataclass
class InMemoryEpisodicMemoryCapability:
    """Contract fake for SESSION recall; keyed by (tenant_id, session_id)."""

    turns: dict[tuple[str, str], list[tuple[str, str]]] = field(default_factory=dict)
    include_cross_session: bool = False
    owner_user_by_session: dict[tuple[str, str], str] = field(default_factory=dict)

    def seed_turn(self, tenant_id: str, session_id: str, turn_id: str, content: str, user_id: str) -> None:
        key = (tenant_id, session_id)
        self.turns.setdefault(key, []).append((turn_id, content))
        self.owner_user_by_session[key] = user_id

    async def recall_session_turns(
        self,
        *,
        tenant_id: str,
        session_id: str,
        query: str,
        top_k: int,
    ) -> tuple[MemoryControlRecallItem, ...]:
        needle = query.strip().lower()
        items: list[MemoryControlRecallItem] = []
        sessions = [(tenant_id, session_id)]
        if self.include_cross_session:
            sessions = [
                key for key in self.turns if key[0] == tenant_id
            ]
        for t_id, s_id in sessions:
            for turn_id, content in self.turns.get((t_id, s_id), ()):
                if needle and needle not in content.lower():
                    continue
                items.append(
                    MemoryControlRecallItem(
                        entry_id=turn_id,
                        content=content,
                        kind=MemoryKind.OTHER,
                        score=1.0,
                    )
                )
                if len(items) >= top_k:
                    break
            if len(items) >= top_k:
                break
        return tuple(items[:top_k])


def build_task_control_plane(
    task_store: InMemoryTaskMemoryCapability | None = None,
) -> MemoryControlPlane:
    store = task_store or InMemoryTaskMemoryCapability()
    return build_default_memory_control_plane(task_memory=store)


def build_session_control_plane(
    episodic: InMemoryEpisodicMemoryCapability,
) -> MemoryControlPlane:
    return build_default_memory_control_plane(episodic=episodic)
