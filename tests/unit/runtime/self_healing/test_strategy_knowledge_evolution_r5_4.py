# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.4 strategy knowledge evolution foundation."""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.knowledge_evolution import (
    KnowledgeEvolutionProcessor,
    SelfHealingWorkflowCompleted,
    StrategyKnowledgeConfidenceLevel,
    StrategyKnowledgeContext,
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionTrigger,
    StrategyKnowledgeProfile,
    StrategyKnowledgeRepository,
    StrategyLearningEngine,
    StrategyMetricProvider,
    mint_strategy_knowledge_profile_id,
)
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
    StrategyKnowledgeRevisionQuery,
)
from intergrax.contracts.self_healing.performance_memory import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
    StrategyPerformanceMemoryRepository,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.runtime.self_healing.knowledge_evolution import (
    BasicStrategyLearningEngine,
    InMemoryStrategyKnowledgeRepository,
    StrategyKnowledgeEvolutionService,
    SuccessRateMetricProvider,
    WorkflowCompletedKnowledgeEvolutionContextBuilder,
    WorkflowCompletedKnowledgeEvolutionProcessor,
)
from intergrax.runtime.self_healing.performance_memory import InMemoryStrategyPerformanceMemoryRepository

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_STRATEGY = "platform.database.reconnect"
_CONTEXT_FP = "diag:db_timeout:v1"
_CONTEXT_REFS = ("investigation://inv-1",)


def _experience(
    outcome: SelfHealingStrategyExecutionOutcome = SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED,
    *,
    suffix: str = "1",
    recorded_at: datetime | None = None,
) -> SelfHealingStrategyPerformanceExperience:
    return SelfHealingStrategyPerformanceExperience(
        experience_id=mint_self_healing_strategy_performance_experience_id(),
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        workflow_id="sh_wf_ke000000001",
        plan_id="sh_plan_ke0000001",
        execution_ids=("exec-1",),
        diagnostic_investigation_id="inv-1",
        execution_outcome=outcome,
        rollback_executed=outcome is SelfHealingStrategyExecutionOutcome.ROLLED_BACK,
        recovery_time_seconds=2.5,
        evidence_refs=(f"evidence://run/{suffix}",),
        recorded_at=recorded_at or datetime(2026, 6, 1, tzinfo=timezone.utc),
    )


def _knowledge_context() -> StrategyKnowledgeContext:
    return StrategyKnowledgeContext(
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        context_fingerprint=_CONTEXT_FP,
        context_refs=_CONTEXT_REFS,
    )


def _evolution_context() -> StrategyKnowledgeEvolutionContext:
    return StrategyKnowledgeEvolutionContext(
        knowledge_context=_knowledge_context(),
        trigger=StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST,
        trigger_refs=("operator://run-1",),
    )


def _service(
    memory: InMemoryStrategyPerformanceMemoryRepository | None = None,
    knowledge: InMemoryStrategyKnowledgeRepository | None = None,
) -> StrategyKnowledgeEvolutionService:
    return StrategyKnowledgeEvolutionService(
        performance_memory=memory or InMemoryStrategyPerformanceMemoryRepository(),
        knowledge_repository=knowledge or InMemoryStrategyKnowledgeRepository(),
        learning_engine=BasicStrategyLearningEngine(),
        metric_provider=SuccessRateMetricProvider(),
    )


def test_plugin_protocols_are_runtime_checkable() -> None:
    assert isinstance(BasicStrategyLearningEngine(), StrategyLearningEngine)
    assert isinstance(SuccessRateMetricProvider(), StrategyMetricProvider)
    assert isinstance(InMemoryStrategyKnowledgeRepository(), StrategyKnowledgeRepository)


def test_profile_id_format_validation() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        StrategyKnowledgeProfile(
            profile_id="bad",
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
            context_refs=_CONTEXT_REFS,
            observation_summary=_observation_summary(),
            quality_snapshot=None,
            confidence_label=StrategyKnowledgeConfidenceLevel.LOW,
            freshness=_freshness(),
            knowledge_version=1,
            supersedes_version=None,
            derived_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            learning_engine_id="test",
            input_experience_fingerprint="sh_exp_fp_empty",
        )


def _observation_summary() -> object:
    from intergrax.contracts.self_healing.knowledge_evolution import StrategyKnowledgeObservationSummary

    return StrategyKnowledgeObservationSummary(
        experience_count=1,
        experience_id_sample=("sh_spm_abc",),
        earliest_recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        latest_recorded_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        evidence_refs=("evidence://1",),
    )


def _freshness() -> object:
    from intergrax.contracts.self_healing.knowledge_evolution import StrategyKnowledgeFreshness

    return StrategyKnowledgeFreshness(
        last_evidence_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        staleness_policy_id=None,
        ttl_hint_seconds=None,
    )


def test_empty_experiences_yields_no_change() -> None:
    result = _service().evolve(_evolution_context())
    assert result.no_change
    assert result.proposed_profile is None


def test_evolve_creates_versioned_profile() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    memory.append(_experience(suffix="2"))
    knowledge = InMemoryStrategyKnowledgeRepository()
    result = _service(memory, knowledge).evolve(_evolution_context())
    assert not result.no_change
    assert result.proposed_profile is not None
    assert result.proposed_profile.knowledge_version == 1
    assert result.proposed_profile.strategy_id == _STRATEGY
    latest = knowledge.get_latest_profile(
        StrategyKnowledgeProfileQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert latest is not None
    assert latest.knowledge_version == 1


def test_idempotent_evolve_skips_second_revision() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    service = _service(memory, knowledge)
    first = service.evolve(_evolution_context())
    second = service.evolve(_evolution_context())
    assert not first.no_change
    assert second.no_change
    revisions = knowledge.list_revisions(
        StrategyKnowledgeRevisionQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert len(revisions) == 1


def test_new_experience_increments_knowledge_version() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience(suffix="1"))
    service = _service(memory, InMemoryStrategyKnowledgeRepository())
    service.evolve(
        StrategyKnowledgeEvolutionContext(
            knowledge_context=_knowledge_context(),
            trigger=StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST,
            trigger_refs=("operator://run-1",),
        ),
    )
    memory.append(_experience(suffix="2", recorded_at=datetime(2026, 6, 2, tzinfo=timezone.utc)))
    second = service.evolve(
        StrategyKnowledgeEvolutionContext(
            knowledge_context=_knowledge_context(),
            trigger=StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST,
            trigger_refs=("operator://run-2",),
        ),
    )
    assert second.proposed_profile is not None
    assert second.proposed_profile.knowledge_version == 2
    assert second.proposed_profile.supersedes_version == 1


def test_same_trigger_skips_second_evolution() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    service = _service(memory, knowledge)
    context = StrategyKnowledgeEvolutionContext(
        knowledge_context=_knowledge_context(),
        trigger=StrategyKnowledgeEvolutionTrigger.WORKFLOW_COMPLETED,
        trigger_refs=("sh_wf_dup", "sh_plan_dup", "sh_spm_x"),
    )
    first = service.evolve(context)
    memory.append(_experience(suffix="extra"))
    second = service.evolve(context)
    assert first.proposed_revision is not None
    assert second.no_change
    revisions = knowledge.list_revisions(
        StrategyKnowledgeRevisionQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert len(revisions) == 1


def test_service_uses_performance_memory_port_only() -> None:
    class RecordingMemory:
        def __init__(self) -> None:
            self.queries: list[tuple[str, str | None]] = []

        def append(
            self,
            experience: SelfHealingStrategyPerformanceExperience,
        ) -> SelfHealingStrategyPerformanceExperience:
            return experience

        def query(self, criteria: object) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
            from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery

            assert isinstance(criteria, StrategyPerformanceMemoryQuery)
            self.queries.append((criteria.tenant_id, criteria.strategy_id))
            return ()

    memory = RecordingMemory()
    _service(memory).evolve(_evolution_context())
    assert memory.queries == [(_TENANT, _STRATEGY)]
    assert isinstance(memory, StrategyPerformanceMemoryRepository)


def test_workflow_processor_is_idempotent() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    exp = _experience()
    memory.append(exp)
    service = _service(memory, InMemoryStrategyKnowledgeRepository())
    processor = WorkflowCompletedKnowledgeEvolutionProcessor(
        evolution_service=service,
        context_builder=WorkflowCompletedKnowledgeEvolutionContextBuilder(),
    )
    event = SelfHealingWorkflowCompleted(
        tenant_id=_TENANT,
        workflow_id="sh_wf_ke000000001",
        strategy_id=_STRATEGY,
        plan_id="sh_plan_ke0000001",
        diagnostic_investigation_id="inv-1",
        context_fingerprint=_CONTEXT_FP,
        context_refs=_CONTEXT_REFS,
        experience_ids=(exp.experience_id,),
        completed_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert isinstance(processor, KnowledgeEvolutionProcessor)
    first = processor.process_workflow_completed(event)
    second = processor.process_workflow_completed(event)
    assert first is not None
    assert second is None


def test_knowledge_evolution_contracts_have_no_runtime_imports() -> None:
    modules = (
        "intergrax.contracts.self_healing.knowledge_evolution.profile",
        "intergrax.contracts.self_healing.knowledge_evolution.engine",
        "intergrax.contracts.self_healing.knowledge_evolution.repository",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path


def test_knowledge_evolution_runtime_has_no_execution_coupling() -> None:
    forbidden_tokens = (
        "lifecycle",
        "orchestrator",
        "SelfHealingActionProvider",
        "execution_engine",
        "HighestConfidenceStrategySelector",
    )
    module_names = (
        "intergrax.runtime.self_healing.knowledge_evolution.service",
        "intergrax.runtime.self_healing.knowledge_evolution.basic_learning_engine",
        "intergrax.runtime.self_healing.knowledge_evolution.processor",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden_tokens:
            assert token.lower() not in lowered


def test_profile_mint_helper() -> None:
    profile_id = mint_strategy_knowledge_profile_id()
    assert profile_id.startswith("sh_skp_")
