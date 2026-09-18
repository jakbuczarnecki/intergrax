# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    SessionTurnIndexStoreQualificationCheck,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.memory_vector_errors import MemoryTenantScopeViolationError
from intergrax.memory.provider_qualification.checks._helpers import failed, passed
from intergrax.memory.provider_qualification.checks._suite import validate_canonical_check_suite

_CAPABILITY = MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _bound_tenant(context: MemoryProviderQualificationContext) -> str:
    """Primary tenant for bound vector adapters and reference stores."""
    return context.tenant_qualification_id


def _other_tenant(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


@dataclass(frozen=True, slots=True)
class SessionTurnIndexTenantIsolationCheck:
    check_id: str = "session_turn_index.tenant_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        marker = f"tenant-marker-{context.qualification_run_id}"
        message = ChatMessage(role="user", content=marker, entry_id=f"entry-tenant-{context.qualification_run_id}")
        await store.upsert_turn(
            tenant_id=_bound_tenant(context),
            session_id=f"session-{context.qualification_run_id}",
            user_id=context.user_qualification_id,
            message=message,
        )
        try:
            hits = await store.search_turns(
                query=marker,
                tenant_id=_other_tenant(context),
                session_id=f"session-{context.qualification_run_id}",
                user_id=context.user_qualification_id,
            )
        except MemoryTenantScopeViolationError:
            hits = []
        if hits:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class SessionTurnIndexSessionIsolationCheck:
    check_id: str = "session_turn_index.session_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        marker = f"session-marker-{context.qualification_run_id}"
        message = ChatMessage(
            role="user",
            content=marker,
            entry_id=f"entry-session-{context.qualification_run_id}",
        )
        session_a = f"session-a-{context.qualification_run_id}"
        session_b = f"session-b-{context.qualification_run_id}"
        await store.upsert_turn(
            tenant_id=_bound_tenant(context),
            session_id=session_a,
            user_id=context.user_qualification_id,
            message=message,
        )
        hits = await store.search_turns(
            query=marker,
            tenant_id=_bound_tenant(context),
            session_id=session_b,
            user_id=context.user_qualification_id,
        )
        if hits:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class SessionTurnIndexUpsertSearchCheck:
    check_id: str = "session_turn_index.upsert_search"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        content = f"searchable-{context.qualification_run_id}"
        entry_id = f"entry-search-{context.qualification_run_id}"
        session_id = f"session-search-{context.qualification_run_id}"
        await store.upsert_turn(
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=context.user_qualification_id,
            message=ChatMessage(role="assistant", content=content, entry_id=entry_id),
        )
        hits = await store.search_turns(
            query=content,
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=context.user_qualification_id,
        )
        if not any(hit.entry_id == entry_id for hit in hits):
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class SessionTurnIndexTombstoneCheck:
    check_id: str = "session_turn_index.tombstone_scope"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        content = f"tombstone-{context.qualification_run_id}"
        entry_id = f"entry-tomb-{context.qualification_run_id}"
        session_id = f"session-tomb-{context.qualification_run_id}"
        await store.upsert_turn(
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=context.user_qualification_id,
            message=ChatMessage(role="user", content=content, entry_id=entry_id),
        )
        await store.tombstone_turn(entry_id)
        hits = await store.search_turns(
            query=content,
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=context.user_qualification_id,
        )
        if hits:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.DELETE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class SessionTurnIndexUserIsolationCheck:
    check_id: str = "session_turn_index.user_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        marker = f"user-marker-{context.qualification_run_id}"
        entry_id = f"entry-user-{context.qualification_run_id}"
        session_id = f"session-user-{context.qualification_run_id}"
        other_user = f"{context.user_qualification_id}-other"
        await store.upsert_turn(
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=context.user_qualification_id,
            message=ChatMessage(role="user", content=marker, entry_id=entry_id),
        )
        hits = await store.search_turns(
            query=marker,
            tenant_id=_bound_tenant(context),
            session_id=session_id,
            user_id=other_user,
        )
        if hits:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class SessionTurnIndexSearchResultFidelityCheck:
    check_id: str = "session_turn_index.search_result_fidelity"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        content = f"fidelity-{context.qualification_run_id}"
        entry_id = f"entry-fidelity-{context.qualification_run_id}"
        session_id = f"session-fidelity-{context.qualification_run_id}"
        tenant_id = _bound_tenant(context)
        message = ChatMessage(role="assistant", content=content, entry_id=entry_id)
        await store.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=context.user_qualification_id,
            message=message,
        )
        hits = await store.search_turns(
            query=content,
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=context.user_qualification_id,
        )
        matched = next((hit for hit in hits if hit.entry_id == entry_id), None)
        if matched is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        if matched.tenant_id != tenant_id:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        if matched.session_id != session_id:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        if matched.user_id != context.user_qualification_id:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        if matched.message.entry_id != entry_id or matched.message.content != content:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


SESSION_TURN_INDEX_STORE_CHECKS: tuple[SessionTurnIndexStoreQualificationCheck, ...] = (
    SessionTurnIndexTenantIsolationCheck(),
    SessionTurnIndexSessionIsolationCheck(),
    SessionTurnIndexUserIsolationCheck(),
    SessionTurnIndexUpsertSearchCheck(),
    SessionTurnIndexTombstoneCheck(),
    SessionTurnIndexSearchResultFidelityCheck(),
)

validate_canonical_check_suite(
    SESSION_TURN_INDEX_STORE_CHECKS,
    capability=_CAPABILITY,
    check_id_of=lambda item: item.check_id,
    capability_of=lambda item: item.capability,
    severity_of=lambda item: item.severity,
)


def default_session_turn_index_checks() -> tuple[SessionTurnIndexStoreQualificationCheck, ...]:
    return SESSION_TURN_INDEX_STORE_CHECKS
