# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-11: hard platform memory boundary guards."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.applications._shared.entity_graph_wiring import resolve_entity_temporal_memory_capability
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityMemoryScope,
    EntityTemporalMemoryCapability,
)
from intergrax.memory.contracts.long_horizon_memory import CanonicalMemorySourceAuthority
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlGovernanceDenied,
    MemoryControlPlane,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    MemoryControlRememberResult,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemorySecurityStrategySet,
)
from intergrax.memory.contracts.procedural_memory import ProcedureMemoryCapability
from intergrax.memory.entity_graph_memory import EntityGraphLegacyBypassError, EntityGraphMemoryStore
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import InMemoryEntityTemporalMemoryStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.strategies import MemoryStrategySet
from intergrax.memory.strategies.defaults.accept_all_promotion import AcceptAllMemoryPromotionStrategy
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.runtime.user_profile.session_memory_consolidation_service import SessionMemoryConsolidationService
from intergrax.applications._shared.canonical_memory_governance_wiring import (
    resolve_canonical_memory_governance_source_authority,
)
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceEntryReader,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
)
from intergrax.memory.memory_security_governance_service import build_default_memory_security_governance_service
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]


def _deny_remember_governance() -> MemorySecurityGovernanceService:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="test.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    class DenyAllAuthorization:
        policy_id = "test.deny"
        policy_version = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            return denied

    strategies = build_default_memory_security_strategy_set()
    custom = MemorySecurityStrategySet(
        authorization=DenyAllAuthorization(),
        trust=strategies.trust,
        admission=strategies.admission,
        governance=strategies.governance,
        retention=strategies.retention,
    )
    return MemorySecurityGovernanceService(strategies=custom)


class _EmptyGovernanceReader(CanonicalMemoryGovernanceEntryReader):
    def list_entries(self, scope):  # type: ignore[no-untyped-def]
        return ()


class _FixedSourceAuthority(CanonicalMemorySourceAuthority):
    def resolve_sources(self, scope, request):  # type: ignore[no-untyped-def]
        return ()


@dataclass(frozen=True, slots=True)
class EntityIndexCall:
    identity: RequestIdentity
    scope: EntityMemoryScope
    entry: UserProfileMemoryEntry


@dataclass
class RecordingEntityMemoryIndexer:
    calls: list[EntityIndexCall] = field(default_factory=list)

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.calls.append(EntityIndexCall(identity=identity, scope=scope, entry=entry))

    def remove_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        memory_entry_id: str,
    ) -> None:
        return None


@dataclass(frozen=True, slots=True)
class RememberCall:
    identity: RequestIdentity
    scope: MemoryControlScopeRef
    request: MemoryControlRememberRequest


@dataclass
class RecordingMemoryControlPlane:
    delegate: MemoryControlPlane
    remember_calls: list[RememberCall] = field(default_factory=list)

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        self.remember_calls.append(
            RememberCall(identity=identity, scope=scope, request=request)
        )
        return await self.delegate.remember(identity, scope, request)


class FailBeforePersistMemoryControlPlane:
    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        raise MemoryControlBackendError("remember failed before canonical persistence")


def _consolidation_strategy_set() -> MemoryStrategySet:
    class _PromoteAll:
        strategy_id = "test.promote.all"

        def promote(self, request):  # type: ignore[no-untyped-def]
            from intergrax.memory.strategies import (
                MemoryPromotionAction,
                MemoryPromotionDecision,
                MemoryPromotionResult,
            )

            return MemoryPromotionResult(
                decisions=(
                    MemoryPromotionDecision(
                        entry=UserProfileMemoryEntry(content="fact", kind=MemoryKind.USER_FACT),
                        action=MemoryPromotionAction.PROMOTE,
                    ),
                )
            )

    class _ExtractOne:
        strategy_id = "test.extract.one"

        async def extract(self, request):  # type: ignore[no-untyped-def]
            from intergrax.memory.strategies import MemoryCandidate, MemoryExtractionResult

            return MemoryExtractionResult(
                candidates=(
                    MemoryCandidate(
                        content="fact",
                        kind=MemoryKind.USER_FACT,
                        session_id=request.session_id,
                    ),
                )
            )

    class _KeepAll:
        strategy_id = "test.dedup.keep"

        def deduplicate(self, request):  # type: ignore[no-untyped-def]
            from intergrax.memory.strategies import MemoryDeduplicationResult

            return MemoryDeduplicationResult(accepted=request.incoming, rejected_as_duplicate=())

    return MemoryStrategySet(
        extraction=_ExtractOne(),
        deduplication=_KeepAll(),
        promotion=_PromoteAll(),
    )


def test_entity_graph_legacy_neighbors_blocks_ungoverned_read() -> None:
    facade = EntityGraphMemoryStore(backend=InMemoryEntityTemporalMemoryStore())
    with pytest.raises(EntityGraphLegacyBypassError):
        facade.neighbors("entity-1", as_of=datetime.now(timezone.utc))


def test_entity_graph_neighbors_parameter_is_datetime_optional() -> None:
    signature = inspect.signature(EntityGraphMemoryStore.neighbors)
    as_of = signature.parameters["as_of"]
    assert as_of.annotation is not object
    assert "datetime" in str(as_of.annotation)


def test_recording_entity_memory_indexer_matches_entity_memory_indexer_protocol() -> None:
    indexer = RecordingEntityMemoryIndexer()
    assert isinstance(indexer, EntityMemoryIndexer)


def test_entity_graph_legacy_facade_blocks_ungoverned_mutations() -> None:
    facade = EntityGraphMemoryStore(backend=InMemoryEntityTemporalMemoryStore())
    with pytest.raises(EntityGraphLegacyBypassError):
        facade.upsert_node(
            __import__(
                "intergrax.memory.entity_graph_memory",
                fromlist=["EntityNode"],
            ).EntityNode(entity_id="e1", label="x"),
        )


def test_memory_platform_wiring_exposes_governed_entity_capability_not_legacy_facade() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.entity_temporal_memory_capability is not None
    assert not isinstance(wiring.entity_temporal_memory_capability, EntityGraphMemoryStore)


def test_session_consolidation_service_does_not_call_profile_manager_add_memory_entry() -> None:
    source = inspect.getsource(SessionMemoryConsolidationService.consolidate_session)
    assert "add_memory_entry" not in source


@pytest.mark.asyncio
async def test_session_consolidation_governance_denial_blocks_writes() -> None:
    tenant_id = "tenant-11"
    manager = UserProfileManager(InMemoryUserProfileStore(), tenant_id=tenant_id)
    plane = build_default_memory_control_plane(
        user_profile_manager=manager,
        security_governance=_deny_remember_governance(),
    )
    entity_indexer = RecordingEntityMemoryIndexer()

    service = SessionMemoryConsolidationService(
        profile_manager=manager,
        instructions_service=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        memory_control_plane=plane,
        strategies=_consolidation_strategy_set(),
        entity_memory_indexer=entity_indexer,
    )
    entries = await service.consolidate_session(
        user_id="user-11",
        session_id="sess-11",
        messages=[ChatMessage(role="user", content="hello")],
        tenant_id=tenant_id,
    )
    assert entries == []
    profile = await manager.get_profile("user-11")
    assert profile.memory_entries == []
    assert entity_indexer.calls == []


@pytest.mark.asyncio
async def test_session_consolidation_propagates_identity_to_entity_projection() -> None:
    tenant_id = "tenant-proj"
    user_id = "user-proj"
    manager = UserProfileManager(InMemoryUserProfileStore(), tenant_id=tenant_id)
    delegate_plane = build_default_memory_control_plane(user_profile_manager=manager)
    recording_plane = RecordingMemoryControlPlane(delegate=delegate_plane)
    entity_indexer = RecordingEntityMemoryIndexer()
    expected_identity = RequestIdentity(tenant_id=tenant_id, user_id=user_id)
    expected_scope = EntityMemoryScope(tenant_id=tenant_id, user_id=user_id)

    from intergrax.runtime.user_profile.session_memory_consolidation_service import (
        SessionMemoryConsolidationConfig,
    )

    service = SessionMemoryConsolidationService(
        profile_manager=manager,
        instructions_service=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        memory_control_plane=recording_plane,
        strategies=_consolidation_strategy_set(),
        entity_memory_indexer=entity_indexer,
        config=SessionMemoryConsolidationConfig(regenerate_system_instructions=False),
    )
    stored = await service.consolidate_session(
        user_id=user_id,
        session_id="sess-proj",
        messages=[ChatMessage(role="user", content="hello")],
        tenant_id=tenant_id,
    )
    assert len(stored) == 1
    assert len(recording_plane.remember_calls) == 1
    assert len(entity_indexer.calls) == 1

    remember_call = recording_plane.remember_calls[0]
    index_call = entity_indexer.calls[0]
    assert remember_call.identity.tenant_id == expected_identity.tenant_id
    assert remember_call.identity.user_id == expected_identity.user_id
    assert index_call.identity.tenant_id == expected_identity.tenant_id
    assert index_call.identity.user_id == expected_identity.user_id
    assert index_call.scope == expected_scope
    assert index_call.entry.content == stored[0].content
    assert index_call.entry.entry_id == stored[0].entry_id
    assert stored[0].entry_id is not None


@pytest.mark.asyncio
async def test_session_consolidation_skips_entity_projection_when_remember_fails() -> None:
    tenant_id = "tenant-fail"
    manager = UserProfileManager(InMemoryUserProfileStore(), tenant_id=tenant_id)
    entity_indexer = RecordingEntityMemoryIndexer()
    service = SessionMemoryConsolidationService(
        profile_manager=manager,
        instructions_service=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        memory_control_plane=FailBeforePersistMemoryControlPlane(),
        strategies=_consolidation_strategy_set(),
        entity_memory_indexer=entity_indexer,
    )
    with pytest.raises(MemoryControlBackendError):
        await service.consolidate_session(
            user_id="user-fail",
            session_id="sess-fail",
            messages=[ChatMessage(role="user", content="hello")],
            tenant_id=tenant_id,
        )
    assert entity_indexer.calls == []


def test_shared_governance_instance_across_memory_capabilities() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.enable_long_horizon_memory = True
    shared = build_default_memory_security_governance_service()
    wiring = resolve_memory_platform_wiring(env, security_governance=shared)
    assert wiring.entity_temporal_memory_capability is not None
    reader = _EmptyGovernanceReader()
    governance_authority = resolve_canonical_memory_governance_source_authority(reader)
    wiring_with_caps = resolve_memory_platform_wiring(
        env,
        security_governance=shared,
        governance_source_authority=governance_authority,
        long_horizon_source_authority=_FixedSourceAuthority(),
    )
    procedural = wiring_with_caps.specialized_memory.procedural_memory_capability
    long_horizon = wiring_with_caps.specialized_memory.long_horizon_memory_capability
    assert isinstance(procedural, ProcedureMemoryCapability)
    assert isinstance(wiring.entity_temporal_memory_capability, EntityTemporalMemoryCapability)
    assert long_horizon is not None
    from intergrax.applications._shared.memory_security_governance_wiring import (
        resolve_memory_security_governance_service,
    )

    assert resolve_memory_security_governance_service(security_governance=shared) is shared


def _module_level_imports_from_memory_stores(rel_path: str) -> list[str]:
    path = _REPO / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.memory.stores."
        ):
            imports.append(node.module)
    return imports


def test_runtime_consolidation_module_has_no_direct_memory_store_imports() -> None:
    rel = "intergrax/runtime/user_profile/session_memory_consolidation_service.py"
    assert _module_level_imports_from_memory_stores(rel) == []


@pytest.mark.asyncio
async def test_memory_control_plane_remember_enforces_governance_before_capability() -> None:
    tenant_id = "tenant-guard"
    manager = UserProfileManager(InMemoryUserProfileStore(), tenant_id=tenant_id)
    plane = build_default_memory_control_plane(
        user_profile_manager=manager,
        security_governance=_deny_remember_governance(),
    )
    identity = RequestIdentity(tenant_id=tenant_id, user_id="user-guard")
    scope = user_memory_scope(identity)
    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.remember(identity, scope, MemoryControlRememberRequest(content="blocked"))
