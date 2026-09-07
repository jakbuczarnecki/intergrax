# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 14 — governed autonomous discovery closed-loop proof."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.autonomous_work.work_stage_capability_loop import (
    WorkStageCapabilityDiscoveryLoopCoordinator,
    WorkStageDiscoveryContextProvider,
    WorkStageDiscoveryIterationContext,
    WorkStageObservationProvider,
    WorkStageToolExecutionPort,
    WorkStageToolExecutionRequest,
    WorkStageToolExecutionResult,
)
from intergrax.autonomous_work.work_stage_tool_execution import RuntimeToolInvokerWorkStagePort
from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    FederatedCapabilityCatalog,
    WorkStageCapabilityDiscoveryService,
)
from intergrax.capability_catalog.adapters.skill_governance import SkillProfileGovernanceEvaluator
from intergrax.capability_catalog.adapters.tool_governance import ToolPolicyGovernanceEvaluator
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.work_stage_effective import WorkStageCapabilityDiscoveryEvidence
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryAvailabilityEvidence,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySetConstraintMode,
    CapabilitySkillGovernanceEvidence,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    LogicalIdentityFilter,
    WorkStageCapabilityLoopDisposition,
    WorkStageCapabilityNeed,
    WorkStageCapabilityObservation,
    WorkStageDomainAuthorityKind,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.builder import build_runtime_state_for_tests, canonical_run_id_for_tests

pytestmark = pytest.mark.unit

_TOOL_A = "tool.loop.stage_a"
_TOOL_B = "tool.loop.stage_b"
_TOOL_DENIED = "tool.loop.stage_denied"
_SKILL_ONLY = "skill.loop.stage_only"
_WORK_REF = "work.stage14.reference"
_GOAL = "prove governed rediscovery loop"
_RUN_ID = "run-stage14-reference"

_OFFICIAL_SOURCE = CapabilitySourceIdentity(
    source_id="official.catalog",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
_PRIVATE_SOURCE = CapabilitySourceIdentity(
    source_id="enterprise.private.catalog",
    source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
)
_BUILTIN_SOURCE = CapabilitySourceIdentity(
    source_id="skills.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)


class _EmptyToolInput(BaseModel):
    pass


class _EmptyToolOutput(BaseModel):
    status: str = "ok"


class _OkToolHandler:
    def execute(self, request: ToolExecutionRequest[_EmptyToolInput]) -> _EmptyToolOutput:
        _ = request
        return _EmptyToolOutput()


class _FailingToolHandler:
    def execute(self, request: ToolExecutionRequest[_EmptyToolInput]) -> _EmptyToolOutput:
        _ = request
        raise RuntimeError("intentional tool failure")


class _FailOnceToolHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[_EmptyToolInput]) -> _EmptyToolOutput:
        _ = request
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("intentional first-call failure")
        return _EmptyToolOutput()


class _StaticSource:
    def __init__(self, source_id: str, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._source_id = source_id
        self._entries = entries
        self.read_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        self.read_calls += 1
        return self._entries


class _MutableSource:
    def __init__(self, source_id: str, entries: list[CapabilityCatalogEntry]) -> None:
        self._source_id = source_id
        self._entries = entries
        self.read_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        self.read_calls += 1
        return tuple(self._entries)


class _FailingSource:
    @property
    def source_id(self) -> str:
        return "zzz.failing"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        raise RuntimeError("catalog backend unavailable")


def _entry(
    *,
    kind: CapabilityKind,
    logical_id: str,
    source: CapabilitySourceIdentity,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source),
        display_label=logical_id,
    )


def _enterprise_scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id="org.stage14",
        tenant_id="tenant.stage14",
        application_id="app.stage14",
        mode=CapabilityDiscoveryScopeMode.ENTERPRISE,
    )


def _evaluators() -> tuple[
    AvailabilityPreservingGovernanceEvaluator,
    ToolPolicyGovernanceEvaluator,
    SkillProfileGovernanceEvaluator,
]:
    return (
        AvailabilityPreservingGovernanceEvaluator(),
        ToolPolicyGovernanceEvaluator(),
        SkillProfileGovernanceEvaluator(),
    )


def _governance_context(
    *,
    allowed_tool_ids: tuple[str, ...] = (),
    denied_tool_ids: tuple[str, ...] = (),
    source: CapabilitySourceIdentity = _OFFICIAL_SOURCE,
) -> CapabilityGovernanceContext:
    tool_allowed = tuple(
        CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id=source.source_id,
            source_kind=source.source_kind,
            logical_id=tool_id,
        )
        for tool_id in allowed_tool_ids
    )
    tool_denied = tuple(
        CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id=source.source_id,
            source_kind=source.source_kind,
            logical_id=tool_id,
        )
        for tool_id in denied_tool_ids
    )
    return CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(
            allowed_keys=tool_allowed,
            denied_keys=tool_denied,
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
        skill_evidence=CapabilitySkillGovernanceEvidence(
            enabled_keys=(),
            enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )


def _availability(
    *entries: CapabilityCatalogEntry,
) -> CapabilityDiscoveryAvailabilityEvidence:
    keys = tuple(
        CapabilityIdentityKey.from_discovery_identity(entry.identity) for entry in entries
    )
    return CapabilityDiscoveryAvailabilityEvidence(
        host_available_keys=keys,
        scope_visible_keys=keys,
    )


def _need(
    *,
    stage_reference: str,
    stage_objective: str,
    logical_ids: tuple[str, ...],
    kinds: tuple[CapabilityKind, ...] = (CapabilityKind.TOOL,),
) -> WorkStageCapabilityNeed:
    return WorkStageCapabilityNeed(
        work_reference=_WORK_REF,
        stage_reference=stage_reference,
        goal_objective=_GOAL,
        stage_objective=stage_objective,
        discovery_query=CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            kinds=kinds,
            logical_identity=LogicalIdentityFilter(exact_logical_ids=logical_ids),
        ),
    )


def _tool_registry(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        contract = ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=_EmptyToolInput,
            output_schema=_EmptyToolOutput,
            side_effects=False,
            error_mapping={},
        )
        registry.register(contract, _OkToolHandler())
    return registry


def _tool_registry_with_handlers(
    handlers: dict[str, object],
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id, handler in handlers.items():
        contract = ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=_EmptyToolInput,
            output_schema=_EmptyToolOutput,
            side_effects=False,
            error_mapping={},
        )
        registry.register(contract, handler)
    return registry


class _AllowAllScopePolicy:
    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        _ = agent_id, tool_id
        return True


@dataclass
class _CountingToolExecution(WorkStageToolExecutionPort):
    inner: WorkStageToolExecutionPort
    calls: int = 0

    def execute(self, request: WorkStageToolExecutionRequest) -> WorkStageToolExecutionResult:
        self.calls += 1
        return self.inner.execute(request)


@dataclass
class _ScriptedObservationProvider(WorkStageObservationProvider):
    next_needs: dict[int, WorkStageCapabilityNeed | None] = field(default_factory=dict)
    blocked_summary: str = "governance blocked"

    def observe(
        self,
        *,
        iteration_index: int,
        need: WorkStageCapabilityNeed,
        selected: GovernedCapabilityCandidate | None,
        execution: WorkStageToolExecutionResult | None,
        discovery: WorkStageCapabilityDiscoveryEvidence,
    ) -> WorkStageCapabilityObservation:
        _ = need, discovery
        if execution is None:
            return WorkStageCapabilityObservation(
                execution_succeeded=False,
                outcome_summary=self.blocked_summary,
                next_need=None,
            )
        return WorkStageCapabilityObservation(
            execution_succeeded=execution.success,
            outcome_summary=execution.output_summary,
            next_need=self.next_needs.get(iteration_index),
        )


@dataclass
class _IterationGovernanceProvider(WorkStageDiscoveryContextProvider):
    availability: CapabilityDiscoveryAvailabilityEvidence
    governance_by_iteration: dict[int, CapabilityGovernanceContext]
    default_governance: CapabilityGovernanceContext

    def iteration_context(
        self,
        need: WorkStageCapabilityNeed,
        iteration_index: int,
    ) -> WorkStageDiscoveryIterationContext:
        _ = need
        governance = self.governance_by_iteration.get(
            iteration_index,
            self.default_governance,
        )
        return WorkStageDiscoveryIterationContext(
            availability_evidence=self.availability,
            governance_context=governance,
        )


def _build_coordinator(
    *,
    federated: FederatedCapabilityCatalog,
    registry: ToolRegistry,
    observation: WorkStageObservationProvider,
    context_provider: WorkStageDiscoveryContextProvider,
    max_iterations: int = 10,
) -> tuple[WorkStageCapabilityDiscoveryLoopCoordinator, _CountingToolExecution]:
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    canonical_run_id = canonical_run_id_for_tests(_RUN_ID)
    inner = RuntimeToolInvokerWorkStagePort(invoker, state)
    counting = _CountingToolExecution(inner=inner)
    coordinator = WorkStageCapabilityDiscoveryLoopCoordinator(
        discovery_service=WorkStageCapabilityDiscoveryService(governance_evaluators=_evaluators()),
        federated_catalog=federated,
        context_provider=context_provider,
        tool_execution=counting,
        observation_provider=observation,
        run_id=canonical_run_id,
        max_iterations=max_iterations,
    )
    return coordinator, counting


def _semantic_outcome(
    outcome: object,
) -> tuple[object, ...]:
    from intergrax.autonomous_work.work_stage_capability_loop import (
        WorkStageCapabilityLoopRunOutcome,
    )

    assert isinstance(outcome, WorkStageCapabilityLoopRunOutcome)
    return (
        outcome.result.disposition,
        tuple(
            (
                item.iteration_index,
                item.need.stage_reference,
                item.selected_identity_key.logical_id if item.selected_identity_key else None,
                item.selected_identity_key.source_id if item.selected_identity_key else None,
                item.domain_authority_kind,
                (
                    item.execution_correlation.run_id,
                    item.execution_correlation.step_id,
                    item.execution_correlation.tool_id,
                )
                if item.execution_correlation
                else None,
                item.observation.outcome_summary if item.observation else None,
                item.observation.next_need.stage_reference
                if item.observation and item.observation.next_need
                else None,
            )
            for item in outcome.result.iterations
        ),
        tuple(
            tuple(
                candidate.identity.logical.logical_id
                for candidate in record.effective_set.effective_candidates
            )
            for record in outcome.discovery_records
        ),
    )


def test_happy_closed_loop_two_iterations() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_PRIVATE_SOURCE)
    official = _StaticSource("official.catalog", (tool_a,))
    private = _StaticSource("enterprise.private.catalog", (tool_b,))
    federated = FederatedCapabilityCatalog((official, private))
    registry = _tool_registry(_TOOL_A, _TOOL_B)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    observation = _ScriptedObservationProvider(next_needs={0: need_b, 1: None})
    context_provider = _IterationGovernanceProvider(
        availability=_availability(tool_a, tool_b),
        governance_by_iteration={
            0: _governance_context(allowed_tool_ids=(_TOOL_A,), source=_OFFICIAL_SOURCE),
            1: _governance_context(
                allowed_tool_ids=(_TOOL_B,),
                source=_PRIVATE_SOURCE,
            ),
        },
        default_governance=_governance_context(),
    )
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=context_provider,
    )

    outcome = coordinator.run(need_a)

    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.COMPLETED
    assert len(outcome.result.iterations) == 2
    assert counting.calls == 2
    assert official.read_calls == 2
    assert private.read_calls == 2
    assert outcome.result.iterations[0].selected_identity_key is not None
    assert outcome.result.iterations[0].selected_identity_key.logical_id == _TOOL_A
    assert outcome.result.iterations[0].selected_identity_key.source_id == "official.catalog"
    assert outcome.result.iterations[1].selected_identity_key is not None
    assert outcome.result.iterations[1].selected_identity_key.logical_id == _TOOL_B
    assert outcome.result.iterations[1].selected_identity_key.source_id == (
        "enterprise.private.catalog"
    )
    assert outcome.result.iterations[0].domain_authority_kind is WorkStageDomainAuthorityKind.TOOL
    assert outcome.discovery_records[0] != outcome.discovery_records[1]
    assert outcome.result.iterations[0].observation is not None
    assert outcome.result.iterations[0].observation.next_need is not None
    assert outcome.result.iterations[0].observation.next_need.stage_reference == "stage.summarize"


def test_governance_deny_mid_loop_prevents_second_execution() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a, tool_b))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry(_TOOL_A, _TOOL_B)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    observation = _ScriptedObservationProvider(next_needs={0: need_b})
    context_provider = _IterationGovernanceProvider(
        availability=_availability(tool_a, tool_b),
        governance_by_iteration={
            0: _governance_context(allowed_tool_ids=(_TOOL_A,)),
            1: _governance_context(
                allowed_tool_ids=(),
                denied_tool_ids=(_TOOL_B,),
            ),
        },
        default_governance=_governance_context(),
    )
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=context_provider,
    )

    outcome = coordinator.run(need_a)

    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.BLOCKED
    assert len(outcome.result.iterations) == 2
    assert counting.calls == 1
    assert outcome.result.iterations[1].selected_identity_key is None
    assert outcome.result.iterations[1].execution_correlation is None
    assert outcome.discovery_records[1].effective_set.governed_result.blocked


def test_observe_rediscover_determinism() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a, tool_b))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry(_TOOL_A, _TOOL_B)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    observation = _ScriptedObservationProvider(next_needs={0: need_b, 1: None})
    context_provider = _IterationGovernanceProvider(
        availability=_availability(tool_a, tool_b),
        governance_by_iteration={},
        default_governance=_governance_context(allowed_tool_ids=(_TOOL_A, _TOOL_B)),
    )
    coordinator_a, _ = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=context_provider,
    )
    coordinator_b, _ = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=context_provider,
    )

    first = _semantic_outcome(coordinator_a.run(need_a))
    second = _semantic_outcome(coordinator_b.run(need_a))
    assert first == second


def test_fresh_catalog_state_after_execution() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    mutable = _MutableSource("official.catalog", [_entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_A,
        source=_OFFICIAL_SOURCE,
    )])
    federated = FederatedCapabilityCatalog((mutable,))
    registry = _tool_registry(_TOOL_A)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_OFFICIAL_SOURCE)
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )

    class _DynamicObservation(_ScriptedObservationProvider):
        def observe(
            self,
            *,
            iteration_index: int,
            need: WorkStageCapabilityNeed,
            selected: GovernedCapabilityCandidate | None,
            execution: WorkStageToolExecutionResult | None,
            discovery: WorkStageCapabilityDiscoveryEvidence,
        ) -> WorkStageCapabilityObservation:
            if iteration_index == 0 and execution is not None:
                mutable._entries.append(tool_b)
                registry.register(
                    ToolContract(
                        tool_id=_TOOL_B,
                        name=_TOOL_B,
                        description=_TOOL_B,
                        input_schema=_EmptyToolInput,
                        output_schema=_EmptyToolOutput,
                        side_effects=False,
                        error_mapping={},
                    ),
                    _OkToolHandler(),
                )
            return super().observe(
                iteration_index=iteration_index,
                need=need,
                selected=selected,
                execution=execution,
                discovery=discovery,
            )

    observation = _DynamicObservation(next_needs={0: need_b, 1: None})
    context_provider = _IterationGovernanceProvider(
        availability=_availability(tool_a, tool_b),
        governance_by_iteration={},
        default_governance=_governance_context(allowed_tool_ids=(_TOOL_A, _TOOL_B)),
    )
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=context_provider,
    )

    outcome = coordinator.run(need_a)

    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.COMPLETED
    assert mutable.read_calls == 2
    assert counting.calls == 2
    assert outcome.result.iterations[1].selected_identity_key is not None
    assert outcome.result.iterations[1].selected_identity_key.logical_id == _TOOL_B
    assert len(outcome.discovery_records[1].effective_set.effective_candidates) == 1


def test_source_failure_fail_closed() -> None:
    failing = FederatedCapabilityCatalog((_FailingSource(),))
    coordinator, counting = _build_coordinator(
        federated=failing,
        registry=_tool_registry(_TOOL_A),
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=CapabilityDiscoveryAvailabilityEvidence(),
            governance_by_iteration={},
            default_governance=_governance_context(),
        ),
    )

    outcome = coordinator.run(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            logical_ids=(_TOOL_A,),
        ),
    )
    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.UNAVAILABLE
    assert counting.calls == 0


def test_identity_conflict_fail_closed() -> None:
    base = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    conflicting = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_A,
        source=_OFFICIAL_SOURCE,
    )
    conflicting = CapabilityCatalogEntry(
        identity=conflicting.identity,
        provenance=conflicting.provenance,
        display_label="conflicting-label",
    )

    class _ConflictingSource:
        read_calls = 0

        @property
        def source_id(self) -> str:
            return "official.catalog"

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            self.read_calls += 1
            return (base, conflicting)

    federated = FederatedCapabilityCatalog((_ConflictingSource(),))
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=_tool_registry(_TOOL_A),
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(base),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A,)),
        ),
    )

    outcome = coordinator.run(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            logical_ids=(_TOOL_A,),
        ),
    )
    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.CONFLICT
    assert counting.calls == 0


def test_no_match_is_not_unavailable() -> None:
    empty = _StaticSource("official.catalog", ())
    federated = FederatedCapabilityCatalog((empty,))
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=_tool_registry(),
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=CapabilityDiscoveryAvailabilityEvidence(),
            governance_by_iteration={},
            default_governance=_governance_context(),
        ),
    )

    outcome = coordinator.run(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            logical_ids=(_TOOL_A,),
        ),
    )
    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.BLOCKED
    assert counting.calls == 0

    coordinator_unavailable, _ = _build_coordinator(
        federated=FederatedCapabilityCatalog((_FailingSource(),)),
        registry=_tool_registry(),
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=CapabilityDiscoveryAvailabilityEvidence(),
            governance_by_iteration={},
            default_governance=_governance_context(),
        ),
    )
    unavailable = coordinator_unavailable.run(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            logical_ids=(_TOOL_A,),
        ),
    )
    assert unavailable.result.disposition is WorkStageCapabilityLoopDisposition.UNAVAILABLE
    assert outcome.result.disposition is not unavailable.result.disposition


def test_no_registry_mutation_from_loop() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a, tool_b))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry(_TOOL_A, _TOOL_B)
    before = frozenset(tool.contract.tool_id for tool in registry.list())
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    coordinator, _ = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=_ScriptedObservationProvider(next_needs={0: need_b, 1: None}),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a, tool_b),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A, _TOOL_B)),
        ),
    )
    coordinator.run(need_a)
    after = frozenset(tool.contract.tool_id for tool in registry.list())
    assert before == after


def test_skill_not_direct_execution_unit() -> None:
    skill = _entry(kind=CapabilityKind.SKILL, logical_id=_SKILL_ONLY, source=_BUILTIN_SOURCE)
    source = _StaticSource("skills.catalog.builtin", (skill,))
    federated = FederatedCapabilityCatalog((source,))
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=_tool_registry(),
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(skill),
            governance_by_iteration={},
            default_governance=CapabilityGovernanceContext(
                posture=CapabilityGovernancePosture.STRICT,
                tool_evidence=CapabilityToolGovernanceEvidence(
                    allowed_keys=(),
                    denied_keys=(),
                    allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
                ),
                skill_evidence=CapabilitySkillGovernanceEvidence(
                    enabled_keys=(
                        CapabilityIdentityKey(
                            kind=CapabilityKind.SKILL,
                            source_id=_BUILTIN_SOURCE.source_id,
                            source_kind=_BUILTIN_SOURCE.source_kind,
                            logical_id=_SKILL_ONLY,
                        ),
                    ),
                    enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
                ),
            ),
        ),
    )
    outcome = coordinator.run(
        _need(
            stage_reference="stage.skill",
            stage_objective="use skill",
            logical_ids=(_SKILL_ONLY,),
            kinds=(CapabilityKind.SKILL,),
        ),
    )
    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.BLOCKED
    assert counting.calls == 0
    assert outcome.result.iterations[0].domain_authority_kind is WorkStageDomainAuthorityKind.SKILL


def test_max_iterations_exhaustion() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a,))
    federated = FederatedCapabilityCatalog((source,))
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_repeat = _need(
        stage_reference="stage.repeat",
        stage_objective="repeat evidence",
        logical_ids=(_TOOL_A,),
    )
    observation = _ScriptedObservationProvider(
        next_needs={0: need_repeat, 1: need_repeat},
    )
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=_tool_registry(_TOOL_A),
        observation=observation,
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A,)),
        ),
        max_iterations=2,
    )
    outcome = coordinator.run(need_a)
    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.ESCALATED
    assert len(outcome.result.iterations) == 2
    assert counting.calls == 2


def test_a4_recovery_never_self_executes() -> None:
    from intergrax.autonomous_work.capability_acquisition_ports import (
        WorkerCapabilityAuthorityCompatibility,
    )
    from intergrax.contracts.autonomous_work.capability_acquisition import (
        WorkerAutonomyLevel,
    )
    from tests.unit.autonomous_work.test_worker_capability_acquisition import (
        _OPERATION,
        _request as acquisition_request,
        _service as acquisition_service,
        _tool_registry as acquisition_tool_registry,
    )

    class _RequiresAuthorityChange:
        def assess(self, *, worker_instance_id, candidate):
            del worker_instance_id, candidate
            return WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED

    service = acquisition_service(
        tool_registry=acquisition_tool_registry(_OPERATION),
        authority=_RequiresAuthorityChange(),
    )
    result = service.decide(acquisition_request())
    assert result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    assert result.decision is not None
    assert result.decision.autonomy_level is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE
    assert result.decision.selected_candidate is None


def test_stage14_architecture_forbidden_abstractions() -> None:
    root = Path(importlib.import_module("intergrax.autonomous_work").__path__[0])
    targets = (
        root / "work_stage_capability_loop.py",
        root / "work_stage_tool_execution.py",
    )
    forbidden = (
        "UniversalCapabilityEngine",
        "UniversalCapabilityRuntime",
        "UniversalCapabilityExecutor",
        "UniversalWorkerOrchestrator",
        "CapabilityDiscoveryPort",
    )
    forbidden_imports = ("pip", "subprocess")
    forbidden_private_attrs = ("_observability_emitter",)
    for path in targets:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for name in forbidden:
            assert name not in source, f"{path.name} must not define or reference {name}"
        for private_attr in forbidden_private_attrs:
            assert private_attr not in source, (
                f"{path.name} must not access private runtime attribute {private_attr}"
            )
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Attribute) and isinstance(node.attr, str):
                if node.attr.startswith("_"):
                    if isinstance(node.value, ast.Name) and node.value.id == "self":
                        continue
                    raise AssertionError(
                        f"{path.name} must not access private attribute {node.attr!r}",
                    )
        for module in imported:
            for forbidden_import in forbidden_imports:
                assert not (
                    module == forbidden_import or module.startswith(f"{forbidden_import}.")
                ), f"{path.name} imports forbidden installer dependency: {module}"


def test_failed_terminal_execution_not_completed() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a,))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry_with_handlers({_TOOL_A: _FailingToolHandler()})
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=_ScriptedObservationProvider(),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A,)),
        ),
    )

    outcome = coordinator.run(need_a)

    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.ESCALATED
    assert outcome.result.disposition is not WorkStageCapabilityLoopDisposition.COMPLETED
    assert len(outcome.result.iterations) == 1
    assert counting.calls == 1
    iteration = outcome.result.iterations[0]
    assert iteration.observation is not None
    assert iteration.observation.execution_succeeded is False
    assert iteration.observation.next_need is None


def test_failed_execution_then_rediscover() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a, tool_b))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry_with_handlers(
        {_TOOL_A: _FailOnceToolHandler(), _TOOL_B: _OkToolHandler()},
    )
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    observation = _ScriptedObservationProvider(next_needs={0: need_b, 1: None})
    coordinator, counting = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=observation,
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a, tool_b),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A, _TOOL_B)),
        ),
    )

    outcome = coordinator.run(need_a)

    assert outcome.result.disposition is WorkStageCapabilityLoopDisposition.COMPLETED
    assert len(outcome.result.iterations) == 2
    assert counting.calls == 2
    assert outcome.result.iterations[0].observation is not None
    assert outcome.result.iterations[0].observation.execution_succeeded is False
    assert outcome.result.iterations[0].observation.next_need is not None
    assert outcome.result.iterations[1].observation is not None
    assert outcome.result.iterations[1].observation.execution_succeeded is True


def test_run_id_mismatch_fails_before_tool_execution() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a,))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry(_TOOL_A)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    inner = RuntimeToolInvokerWorkStagePort(invoker, state)
    counting = _CountingToolExecution(inner=inner)
    mismatched_run_id = canonical_run_id_for_tests("run-stage14-mismatch-seed")
    coordinator = WorkStageCapabilityDiscoveryLoopCoordinator(
        discovery_service=WorkStageCapabilityDiscoveryService(governance_evaluators=_evaluators()),
        federated_catalog=federated,
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A,)),
        ),
        tool_execution=counting,
        observation_provider=_ScriptedObservationProvider(),
        run_id=mismatched_run_id,
        max_iterations=10,
    )

    with pytest.raises(ValueError, match="does not match RuntimeState.run_id"):
        coordinator.run(
            _need(
                stage_reference="stage.collect",
                stage_objective="collect evidence",
                logical_ids=(_TOOL_A,),
            ),
        )
    assert not any(event.step == "tool_invocation_start" for event in state.trace_events)


def test_execution_correlation_matches_canonical_tool_path() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    source = _StaticSource("official.catalog", (tool_a,))
    federated = FederatedCapabilityCatalog((source,))
    registry = _tool_registry(_TOOL_A)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    coordinator, _ = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=_ScriptedObservationProvider(next_needs={0: None}),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a),
            governance_by_iteration={},
            default_governance=_governance_context(allowed_tool_ids=(_TOOL_A,)),
        ),
    )

    outcome = coordinator.run(need_a)
    iteration = outcome.result.iterations[0]
    assert iteration.execution_correlation is not None
    assert iteration.execution_correlation.tool_id == _TOOL_A
    assert iteration.execution_correlation.step_id == "0"
    assert iteration.execution_correlation.run_id == canonical_run_id_for_tests(_RUN_ID)
    assert iteration.execution_correlation.tool_id == iteration.selected_identity_key.logical_id


def test_happy_closed_loop_records_execution_correlation() -> None:
    tool_a = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_A, source=_OFFICIAL_SOURCE)
    tool_b = _entry(kind=CapabilityKind.TOOL, logical_id=_TOOL_B, source=_PRIVATE_SOURCE)
    official = _StaticSource("official.catalog", (tool_a,))
    private = _StaticSource("enterprise.private.catalog", (tool_b,))
    federated = FederatedCapabilityCatalog((official, private))
    registry = _tool_registry(_TOOL_A, _TOOL_B)
    need_a = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        logical_ids=(_TOOL_A,),
    )
    need_b = _need(
        stage_reference="stage.summarize",
        stage_objective="summarize evidence",
        logical_ids=(_TOOL_B,),
    )
    coordinator, _ = _build_coordinator(
        federated=federated,
        registry=registry,
        observation=_ScriptedObservationProvider(next_needs={0: need_b, 1: None}),
        context_provider=_IterationGovernanceProvider(
            availability=_availability(tool_a, tool_b),
            governance_by_iteration={
                0: _governance_context(allowed_tool_ids=(_TOOL_A,), source=_OFFICIAL_SOURCE),
                1: _governance_context(
                    allowed_tool_ids=(_TOOL_B,),
                    source=_PRIVATE_SOURCE,
                ),
            },
            default_governance=_governance_context(),
        ),
    )

    outcome = coordinator.run(need_a)

    for index, iteration in enumerate(outcome.result.iterations):
        assert iteration.execution_correlation is not None
        assert iteration.execution_correlation.step_id == str(index)
        assert iteration.execution_correlation.run_id == canonical_run_id_for_tests(_RUN_ID)
