# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-11: hard platform memory boundary guards."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.entity_graph_wiring import resolve_entity_temporal_memory_capability
from intergrax.applications._shared.long_horizon_memory_wiring import resolve_long_horizon_memory_capability
from intergrax.applications._shared.procedural_memory_wiring import resolve_procedural_memory_capability
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryCapability
from intergrax.memory.contracts.long_horizon_memory import CanonicalMemorySourceAuthority
from intergrax.memory.contracts.memory_control import MemoryControlGovernanceDenied, MemoryControlRememberRequest, user_memory_scope
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
from intergrax.memory.user_profile_memory import MemoryKind
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

    class _PromoteAll:
        strategy_id = "test.promote.all"

        def promote(self, request):  # type: ignore[no-untyped-def]
            from intergrax.memory.strategies import MemoryPromotionAction, MemoryPromotionDecision, MemoryPromotionResult
            from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

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

    service = SessionMemoryConsolidationService(
        profile_manager=manager,
        instructions_service=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        memory_control_plane=plane,
        strategies=MemoryStrategySet(
            extraction=_ExtractOne(),
            deduplication=_KeepAll(),
            promotion=_PromoteAll(),
        ),
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


def test_shared_governance_instance_across_memory_capabilities() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env.memory_profile.enable_procedural_memory = True
    env.memory_profile.enable_long_horizon_memory = True
    shared = build_default_memory_security_governance_service()
    wiring = resolve_memory_platform_wiring(env, security_governance=shared)
    assert wiring.entity_temporal_memory_capability is not None
    reader = _EmptyGovernanceReader()
    governance_authority = resolve_canonical_memory_governance_source_authority(reader)
    procedural = resolve_procedural_memory_capability(
        env,
        governance_source_authority=governance_authority,
        security_governance=shared,
    )
    long_horizon = resolve_long_horizon_memory_capability(
        env,
        source_authority=_FixedSourceAuthority(),
        governance_source_authority=governance_authority,
        security_governance=shared,
    )
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
