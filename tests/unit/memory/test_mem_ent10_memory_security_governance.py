# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-10: memory security & governance boundary."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import get_type_hints

import pytest

import intergrax.memory.contracts.memory_control as memory_control_module
import intergrax.memory.contracts.memory_security_governance as memory_security_governance_module
import intergrax.memory.memory_security_governance_service as memory_security_governance_service_module
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlGovernanceDenied,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryGovernanceRecordSnapshot,
    MemorySecurityContext,
    MemorySecurityStrategySet,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.strategies.defaults.memory_security_governance import (
    DefaultMemoryAdmissionPolicy,
    DefaultMemoryAuthorizationPolicy,
    DefaultMemoryGovernancePolicy,
    DefaultMemoryRetentionPolicy,
    DefaultMemoryTrustEvaluationPolicy,
    build_default_memory_security_strategy_set,
)
from intergrax.memory.strategies.recall_models import MemoryRecallCandidate, MemoryRetrievalSource
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

pytestmark = pytest.mark.gate

_TENANT = "tenant-10"
_USER = "user-10"


def _identity() -> RequestIdentity:
    return RequestIdentity(tenant_id=_TENANT, user_id=_USER)


def _plane(
    *,
    governance: MemorySecurityGovernanceService | None = None,
) -> DefaultMemoryControlPlane:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store=store)
    return DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        security_governance=governance,
    )


def _entry(
    *,
    content: str = "fact",
    trust: MemoryTrustClass = MemoryTrustClass.USER_EXPLICIT,
    source: MemoryRecordSourceType = MemoryRecordSourceType.USER_EXPLICIT,
    classification: DataClassification = DataClassification.INTERNAL,
) -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        content=content,
        provenance=MemoryProvenance(source_type=source),
        trust=MemoryRecordTrust(trust_class=trust),
        governance=MemoryRecordGovernance(data_classification=classification),
    )


@pytest.mark.asyncio
async def test_allow_write_permits_mutation() -> None:
    plane = _plane()
    scope = user_memory_scope(_identity())
    result = await plane.remember(
        _identity(),
        scope,
        MemoryControlRememberRequest(content="hello governance"),
    )
    assert result.entry_id


@pytest.mark.asyncio
async def test_deny_write_zero_mutation() -> None:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="test.policy",
        policy_version="9",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class DenyAllAuthorization:
        policy_id: str = "test.deny"
        policy_version: str = "1"

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
    service = MemorySecurityGovernanceService(strategies=custom)
    plane = _plane(governance=service)
    scope = user_memory_scope(_identity())
    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.remember(_identity(), scope, MemoryControlRememberRequest(content="blocked"))
    recall = await plane.recall(_identity(), scope, MemoryControlRecallRequest(query=""))
    assert recall.items == ()


@pytest.mark.asyncio
async def test_restricted_recall_filtered() -> None:
    plane = _plane()
    scope = user_memory_scope(_identity())
    public_entry = _entry(content="public fact", classification=DataClassification.INTERNAL)
    restricted = _entry(content="secret", classification=DataClassification.RESTRICTED)
    await plane.remember(_identity(), scope, MemoryControlRememberRequest(entry=public_entry))
    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.remember(_identity(), scope, MemoryControlRememberRequest(entry=restricted))
    recall = await plane.recall(_identity(), scope, MemoryControlRecallRequest(query=""))
    ids = {item.entry_id for item in recall.items}
    assert public_entry.entry_id in ids


@pytest.mark.asyncio
async def test_trust_escalation_blocked_on_write() -> None:
    plane = _plane()
    scope = user_memory_scope(_identity())
    poisoned = _entry(
        trust=MemoryTrustClass.USER_EXPLICIT,
        source=MemoryRecordSourceType.UNKNOWN,
    )
    with pytest.raises(MemoryControlGovernanceDenied) as exc:
        await plane.remember(_identity(), scope, MemoryControlRememberRequest(entry=poisoned))
    assert exc.value.decision.reason_code is MemoryGovernanceReasonCode.POISONING_SUSPECTED


@pytest.mark.asyncio
async def test_recall_mixed_candidates_filters_denied() -> None:
    service = build_default_memory_security_governance_service()
    scope = user_memory_scope(_identity())
    ctx = MemorySecurityContext(
        identity=_identity(),
        scope=scope,
        operation=MemoryGovernanceOperation.RECALL,
    )
    request = MemoryGovernanceEvaluationRequest(context=ctx)
    allowed_entry = _entry(classification=DataClassification.INTERNAL)
    denied_entry = _entry(classification=DataClassification.RESTRICTED)
    candidates = (
        MemoryRecallCandidate(
            record=allowed_entry,
            retrieval_source=MemoryRetrievalSource.PROFILE_SCAN,
            retrieval_score=None,
        ),
        MemoryRecallCandidate(
            record=denied_entry,
            retrieval_source=MemoryRetrievalSource.PROFILE_SCAN,
            retrieval_score=None,
        ),
    )
    filtered = service.filter_recall_candidates(request, candidates)
    assert len(filtered) == 1
    assert filtered[0].record.entry_id == allowed_entry.entry_id


def test_fail_closed_on_policy_exception() -> None:
    class BrokenAdmission:
        policy_id = "broken"
        policy_version = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            raise RuntimeError("boom")

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=base.authorization,
            trust=base.trust,
            admission=BrokenAdmission(),
            governance=base.governance,
            retention=base.retention,
        )
    )
    scope = user_memory_scope(_identity())
    decision = service.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=_identity(),
                scope=scope,
                operation=MemoryGovernanceOperation.REMEMBER,
            ),
            proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(_entry()),
        )
    )
    assert decision.outcome is MemoryGovernanceOutcome.DENY
    assert decision.reason_code is MemoryGovernanceReasonCode.POLICY_FAILURE


def test_missing_strategy_set_fail_closed() -> None:
    service = MemorySecurityGovernanceService(
        strategies=build_default_memory_security_strategy_set(),
    )
    service.strategies = None  # type: ignore[assignment]
    scope = user_memory_scope(_identity())
    decision = service.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=_identity(),
                scope=scope,
                operation=MemoryGovernanceOperation.RECALL,
            ),
        )
    )
    assert decision.outcome is MemoryGovernanceOutcome.DENY


def test_decision_includes_policy_metadata() -> None:
    service = build_default_memory_security_governance_service()
    scope = user_memory_scope(_identity())
    decision = service.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=_identity(),
                scope=scope,
                operation=MemoryGovernanceOperation.REMEMBER,
            ),
            proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(_entry()),
        )
    )
    assert decision.policy_id
    assert decision.policy_version
    assert decision.reason_code is MemoryGovernanceReasonCode.ALLOWED


def test_default_strategies_deterministic() -> None:
    scope = user_memory_scope(_identity())
    request = MemoryGovernanceEvaluationRequest(
        context=MemorySecurityContext(
            identity=_identity(),
            scope=scope,
            operation=MemoryGovernanceOperation.REMEMBER,
        ),
        proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(_entry()),
    )
    service = build_default_memory_security_governance_service()
    first = service.evaluate(request)
    second = service.evaluate(request)
    assert first == second


@pytest.mark.asyncio
async def test_deny_delete_blocks_removal() -> None:
    base_governance = DefaultMemoryGovernancePolicy()

    @dataclass(frozen=True, slots=True)
    class DenyDeleteGovernance:
        policy_id: str = "memory.governance.deny_delete"
        policy_version: str = "1.0.0"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            if request.context.operation is MemoryGovernanceOperation.DELETE:
                return MemoryGovernanceDecision(
                    outcome=MemoryGovernanceOutcome.DENY,
                    reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    operation=request.context.operation,
                )
            return base_governance.evaluate(request)

    strategies = MemorySecurityStrategySet(
        authorization=DefaultMemoryAuthorizationPolicy(),
        trust=DefaultMemoryTrustEvaluationPolicy(),
        admission=DefaultMemoryAdmissionPolicy(),
        governance=DenyDeleteGovernance(),
        retention=DefaultMemoryRetentionPolicy(),
    )
    plane = _plane(governance=MemorySecurityGovernanceService(strategies=strategies))
    scope = user_memory_scope(_identity())
    remembered = await plane.remember(
        _identity(),
        scope,
        MemoryControlRememberRequest(content="keep me"),
    )
    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.forget(
            _identity(),
            scope,
            MemoryControlForgetRequest(entry_id=remembered.entry_id),
        )


def test_no_security_governance_config_bypass_api() -> None:
    assert not hasattr(memory_security_governance_module, "MemorySecurityGovernanceConfig")


def test_policies_always_evaluated_no_disabled_allow_path() -> None:
    service = build_default_memory_security_governance_service()
    scope = user_memory_scope(_identity())
    decision = service.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=_identity(),
                scope=scope,
                operation=MemoryGovernanceOperation.REMEMBER,
            ),
            proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(_entry()),
        )
    )
    assert decision.policy_id != "memory.security.disabled"


def test_custom_deny_authorization_still_deny() -> None:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.AUTHORIZATION_DENY,
        policy_id="test.auth.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class DenyAuthorization:
        policy_id: str = "test.auth.deny"
        policy_version: str = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            return denied

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=DenyAuthorization(),
            trust=base.trust,
            admission=base.admission,
            governance=base.governance,
            retention=base.retention,
        )
    )
    scope = user_memory_scope(_identity())
    decision = service.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=_identity(),
                scope=scope,
                operation=MemoryGovernanceOperation.REMEMBER,
            ),
            proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(_entry()),
        )
    )
    assert decision.outcome is MemoryGovernanceOutcome.DENY
    assert decision.reason_code is MemoryGovernanceReasonCode.AUTHORIZATION_DENY


def test_governance_denied_exception_decision_typed() -> None:
    merged_globals = {
        **vars(memory_control_module),
        "MemoryGovernanceDecision": MemoryGovernanceDecision,
    }
    hints = get_type_hints(
        MemoryControlGovernanceDenied.__init__,
        globalns=merged_globals,
        localns=merged_globals,
    )
    assert hints["decision"] is MemoryGovernanceDecision


@pytest.mark.asyncio
async def test_governance_denied_exception_carries_evaluated_decision() -> None:
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="test.exact",
        policy_version="2",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    @dataclass(frozen=True, slots=True)
    class DenyAllAuthorization:
        policy_id: str = "test.exact"
        policy_version: str = "2"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            return denied

    base = build_default_memory_security_strategy_set()
    service = MemorySecurityGovernanceService(
        strategies=MemorySecurityStrategySet(
            authorization=DenyAllAuthorization(),
            trust=base.trust,
            admission=base.admission,
            governance=base.governance,
            retention=base.retention,
        )
    )
    plane = _plane(governance=service)
    scope = user_memory_scope(_identity())
    with pytest.raises(MemoryControlGovernanceDenied) as exc:
        await plane.remember(_identity(), scope, MemoryControlRememberRequest(content="x"))
    assert exc.value.decision == denied
    assert isinstance(exc.value.decision, MemoryGovernanceDecision)


def test_memory_security_public_contracts_no_object_or_any_annotations() -> None:
    targets = (
        memory_control_module.MemoryControlGovernanceDenied,
        memory_security_governance_module.MemoryGovernanceDecision,
        memory_security_governance_service_module.MemorySecurityGovernanceService,
    )
    forbidden = {"object", "Any"}
    for target in targets:
        tree = ast.parse(inspect.getsource(target))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in forbidden:
                raise AssertionError(f"{target.__name__} must not use literal {node.value!r}")
            if isinstance(node, ast.Name) and node.id in forbidden:
                if isinstance(getattr(node, "ctx", None), ast.Load):
                    raise AssertionError(f"{target.__name__} references forbidden name {node.id!r}")


def test_memory_security_modules_no_private_cross_module_imports() -> None:
    root = memory_security_governance_service_module.__file__
    assert root is not None
    modules = (
        memory_control_module,
        memory_security_governance_module,
        memory_security_governance_service_module,
    )
    for mod in modules:
        path = mod.__file__
        assert path is not None
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.startswith("intergrax.memory.") and "._" in node.module.split(".")[-1]:
                raise AssertionError(f"{path} imports private module {node.module}")
            for alias in node.names:
                if alias.name.startswith("_"):
                    raise AssertionError(f"{path} imports private symbol {node.module}.{alias.name}")
