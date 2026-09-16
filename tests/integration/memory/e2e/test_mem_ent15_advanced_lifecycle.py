# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: procedural promotion projection + long-horizon compaction."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceSnapshot,
    LongHorizonCompactionRequest,
    LongHorizonCompactionSource,
    LongHorizonMemoryScope,
)
from intergrax.memory.contracts.memory_control import MemoryControlRememberRequest
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    procedure_id_for_source_memory,
)
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.memory.memory_security_governance_service import (
    build_default_memory_security_governance_service,
)
from intergrax.memory.procedural_memory_indexing import DefaultProceduralMemoryIndexer
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from tests.integration.memory.e2e.harness import build_in_memory_memory_harness
from tests.unit.memory.governance_source_fixtures import PermissiveCanonicalGovernanceSourceAuthority

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


class _ProfileBackedSourceAuthority:
    def __init__(self, harness_scope: LongHorizonMemoryScope, harness) -> None:
        self._scope = harness_scope
        self._harness = harness

    def resolve_canonical_source(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemorySourceSnapshot:
        if (
            scope.tenant_id != self._scope.tenant_id
            or scope.user_id != self._scope.user_id
        ):
            raise ValueError("scope mismatch")
        return CanonicalMemorySourceSnapshot(
            memory_id=memory_id,
            revision=revision,
            content=f"lh-content:{memory_id}",
            observed_at="2026-03-01T12:00:00+00:00",
        )


@pytest.mark.asyncio
async def test_procedural_promotion_from_canonical_memory_entry() -> None:
    harness = build_in_memory_memory_harness(
        tenant_id="tenant-ent15-proc", user_id="user-proc", include_entity_projection=False
    )
    identity = harness.identity()
    scope = harness.user_scope(identity)
    remembered = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(
            content="run backup weekly",
            kind=MemoryKind.PROCEDURAL,
            title="backup-runbook",
        ),
    )
    assert remembered.entry_id
    profile = await harness.canonical_profile()
    entry = next(e for e in profile.memory_entries if e.entry_id == remembered.entry_id)

    proc_store = InMemoryProceduralMemoryStore()
    governance = build_default_memory_security_governance_service()
    indexer = DefaultProceduralMemoryIndexer(proc_store, security_governance=governance)
    proc_scope = ProceduralMemoryScope(
        tenant_id=harness.tenant_id,
        user_id=harness.user_id,
    )
    record = indexer.index_memory_entry(identity, proc_scope, entry)
    assert record is not None
    pid = procedure_id_for_source_memory(proc_scope, entry.entry_id)
    stored = proc_store.get_procedure(proc_scope, pid)
    assert stored is not None
    assert stored.source_memory_id == entry.entry_id
    assert stored.source_memory_revision == entry.revision


@pytest.mark.asyncio
async def test_long_horizon_compaction_preserves_source_authority() -> None:
    harness = build_in_memory_memory_harness(
        tenant_id="tenant-ent15-lh", user_id="user-lh", include_entity_projection=False
    )
    identity = harness.identity()
    scope = harness.user_scope(identity)
    remembered = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="session fact for compaction", kind=MemoryKind.USER_FACT),
    )
    assert remembered.entry_id
    profile = await harness.canonical_profile()
    entry = next(e for e in profile.memory_entries if e.entry_id == remembered.entry_id)

    lh_scope = LongHorizonMemoryScope(
        tenant_id=harness.tenant_id,
        user_id=harness.user_id,
    )
    lh_store = InMemoryLongHorizonMemoryStore()
    service = LongHorizonMemoryService(
        _store=lh_store,
        _strategies=build_default_long_horizon_strategies(),
        _source_authority=_ProfileBackedSourceAuthority(lh_scope, harness),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
        _security_governance=build_default_memory_security_governance_service(),
    )
    request = LongHorizonCompactionRequest(
        identity=RequestIdentity(
            tenant_id=harness.tenant_id,
            user_id=harness.user_id,
        ),
        scope=lh_scope,
        target_level=1,
        sources=(
            LongHorizonCompactionSource(
                memory_id=entry.entry_id,
                revision=entry.revision,
                content=entry.content or "",
                observed_at="2026-03-01T12:00:00+00:00",
            ),
        ),
    )
    result = service.compact(request)
    assert not result.failures
    assert result.created
    summary = result.created[0]
    assert summary.source_memory_refs
    assert summary.source_memory_refs[0].memory_id == entry.entry_id
    assert summary.source_memory_refs[0].revision == entry.revision
