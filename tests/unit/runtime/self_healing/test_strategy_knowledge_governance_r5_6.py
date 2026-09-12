# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.6 knowledge governance and audit evolution."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.knowledge_evolution import (
    StrategyKnowledgeAuditRepository,
    StrategyKnowledgeChangeRecord,
    StrategyKnowledgeChangeRecordQuery,
    StrategyKnowledgeChangeSource,
    StrategyKnowledgeChangeType,
    StrategyKnowledgeContext,
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionTrigger,
    StrategyKnowledgeGovernancePolicy,
    StrategyKnowledgeIntegrityValidator,
    StrategyKnowledgeUpdated,
    mint_strategy_knowledge_change_id,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.policy import (
    StrategyKnowledgeGovernanceAssessment,
    StrategyKnowledgeGovernanceControlLevel,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision
from intergrax.contracts.self_healing.performance_memory import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.runtime.self_healing.knowledge_evolution import (
    BasicStrategyKnowledgeIntegrityValidator,
    BasicStrategyLearningEngine,
    DefaultKnowledgeGovernancePolicy,
    InMemoryStrategyKnowledgeAuditRepository,
    InMemoryStrategyKnowledgeRepository,
    StrategyKnowledgeEvolutionService,
    StrategyKnowledgeGovernanceService,
    SuccessRateMetricProvider,
)
from intergrax.runtime.self_healing.knowledge_evolution.governance.change_record_builder import (
    build_change_record_from_revision,
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
        recorded_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
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
        trigger=StrategyKnowledgeEvolutionTrigger.WORKFLOW_COMPLETED,
        trigger_refs=("workflow://sh_wf_ke000000001",),
    )


def _governance_service(
    audit: InMemoryStrategyKnowledgeAuditRepository | None = None,
) -> StrategyKnowledgeGovernanceService:
    return StrategyKnowledgeGovernanceService(
        audit_repository=audit or InMemoryStrategyKnowledgeAuditRepository(),
        governance_policy=DefaultKnowledgeGovernancePolicy(),
        integrity_validator=BasicStrategyKnowledgeIntegrityValidator(),
    )


def _evolution_service(
    memory: InMemoryStrategyPerformanceMemoryRepository,
    knowledge: InMemoryStrategyKnowledgeRepository,
    governance: StrategyKnowledgeGovernanceService | None = None,
) -> StrategyKnowledgeEvolutionService:
    return StrategyKnowledgeEvolutionService(
        performance_memory=memory,
        knowledge_repository=knowledge,
        learning_engine=BasicStrategyLearningEngine(),
        metric_provider=SuccessRateMetricProvider(),
        knowledge_governance=governance,
    )


def test_change_record_from_revision_links_versions_and_experiences() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    governance = _governance_service()
    service = _evolution_service(memory, knowledge, governance)
    result = service.evolve(_evolution_context())
    assert result.proposed_revision is not None
    change = build_change_record_from_revision(result.proposed_revision)
    assert change.change_type is StrategyKnowledgeChangeType.INITIAL_PROFILE
    assert change.change_source is StrategyKnowledgeChangeSource.WORKFLOW_COMPLETED
    assert change.previous_knowledge_version is None
    assert change.new_knowledge_version == 1
    assert change.source_experience_refs


def test_evolution_with_governance_writes_audit_trail() -> None:
    from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_query import (
        StrategyKnowledgeUpdatedEventQuery,
    )

    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    audit = InMemoryStrategyKnowledgeAuditRepository()
    governance = _governance_service(audit)
    service = _evolution_service(memory, knowledge, governance)
    service.evolve(_evolution_context())
    records = audit.list_change_records(
        StrategyKnowledgeChangeRecordQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert len(records) == 1
    events = audit.list_knowledge_updated_events(
        StrategyKnowledgeUpdatedEventQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert len(events) == 1


def test_evolution_with_governance_emits_knowledge_updated_event() -> None:
    from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_query import (
        StrategyKnowledgeUpdatedEventQuery,
    )

    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    audit = InMemoryStrategyKnowledgeAuditRepository()
    governance = _governance_service(audit)
    service = _evolution_service(memory, InMemoryStrategyKnowledgeRepository(), governance)
    service.evolve(_evolution_context())
    events = audit.list_knowledge_updated_events(
        StrategyKnowledgeUpdatedEventQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert len(events) == 1
    assert isinstance(events[0], StrategyKnowledgeUpdated)
    assert events[0].new_knowledge_version == 1


def test_integrity_validator_passes_when_audit_aligned() -> None:
    from intergrax.contracts.self_healing.knowledge_evolution.query import StrategyKnowledgeProfileQuery

    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    audit = InMemoryStrategyKnowledgeAuditRepository()
    governance = _governance_service(audit)
    service = _evolution_service(memory, knowledge, governance)
    service.evolve(_evolution_context())
    validator = BasicStrategyKnowledgeIntegrityValidator()
    report = validator.validate_scope(
        knowledge_repository=knowledge,
        audit_repository=audit,
        scope=StrategyKnowledgeProfileQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert report.is_valid


def test_default_governance_policy_is_record_only() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    audit = InMemoryStrategyKnowledgeAuditRepository()
    governance = _governance_service(audit)
    service = _evolution_service(memory, InMemoryStrategyKnowledgeRepository(), governance)
    result = service.evolve(_evolution_context())
    assert result.proposed_revision is not None
    records = audit.list_change_records(
        StrategyKnowledgeChangeRecordQuery(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
        ),
    )
    assert records
    assessment = DefaultKnowledgeGovernancePolicy().assess_change(
        records[0],
        result.proposed_revision,
    )
    assert assessment.control_level is StrategyKnowledgeGovernanceControlLevel.RECORD_ONLY


def test_governance_contracts_are_swappable() -> None:
    @dataclass
    class AlternateAuditRepository:
        stored: StrategyKnowledgeChangeRecord | None = None

        def append_change_record(
            self,
            record: StrategyKnowledgeChangeRecord,
        ) -> StrategyKnowledgeChangeRecord:
            self.stored = record
            return record

        def list_change_records(
            self,
            criteria: StrategyKnowledgeChangeRecordQuery,
        ) -> tuple[StrategyKnowledgeChangeRecord, ...]:
            _ = criteria
            return (self.stored,) if self.stored is not None else ()

        def append_knowledge_updated_event(
            self,
            event: StrategyKnowledgeUpdated,
        ) -> StrategyKnowledgeUpdated:
            return event

        def list_knowledge_updated_events(
            self,
            criteria: object,
        ) -> tuple[StrategyKnowledgeUpdated, ...]:
            _ = criteria
            return ()

    @dataclass(frozen=True, slots=True)
    class AlternateGovernancePolicy:
        _policy_id: str = "test.alternate_governance"

        @property
        def policy_id(self) -> str:
            return self._policy_id

        def assess_change(
            self,
            change: StrategyKnowledgeChangeRecord,
            revision: StrategyKnowledgeRevision,
        ) -> StrategyKnowledgeGovernanceAssessment:
            _ = revision
            return StrategyKnowledgeGovernanceAssessment(
                policy_id=self.policy_id,
                control_level=StrategyKnowledgeGovernanceControlLevel.ENHANCED_AUDIT,
                audit_tags=(change.change_id,),
                rationale="alternate policy assessment",
            )

    audit = AlternateAuditRepository()
    governance = StrategyKnowledgeGovernanceService(
        audit_repository=audit,
        governance_policy=AlternateGovernancePolicy(),
    )
    assert isinstance(audit, StrategyKnowledgeAuditRepository)
    assert isinstance(governance.governance_policy, StrategyKnowledgeGovernancePolicy)

    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    knowledge = InMemoryStrategyKnowledgeRepository()
    service = _evolution_service(memory, knowledge, governance)
    result = service.evolve(_evolution_context())
    assert result.proposed_revision is not None
    assert audit.stored is not None
    assessment = governance.governance_policy.assess_change(audit.stored, result.proposed_revision)
    assert assessment.policy_id == "test.alternate_governance"


def test_change_record_model_rejects_invalid_versions() -> None:
    with pytest.raises(ValueError, match="previous_knowledge_version"):
        StrategyKnowledgeChangeRecord(
            change_id=mint_strategy_knowledge_change_id(),
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            context_fingerprint=_CONTEXT_FP,
            revision_id="sh_skr_deadbeef",
            previous_knowledge_version=2,
            new_knowledge_version=1,
            change_source=StrategyKnowledgeChangeSource.BACKFILL,
            change_type=StrategyKnowledgeChangeType.VERSION_INCREMENT,
            evolution_mechanism_id="engine",
            source_experience_refs=(),
            rationale="invalid",
            recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_governance_domain_contracts_have_no_runtime_imports() -> None:
    modules = (
        "intergrax.contracts.self_healing.knowledge_evolution.governance.change_record",
        "intergrax.contracts.self_healing.knowledge_evolution.governance.policy",
        "intergrax.contracts.self_healing.knowledge_evolution.governance.audit_repository",
        "intergrax.contracts.self_healing.knowledge_evolution.governance.integrity",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path


def test_governance_runtime_has_no_execution_coupling() -> None:
    forbidden_tokens = (
        "lifecycle",
        "orchestrator",
        "SelfHealingActionProvider",
        "execution_engine",
        "HighestConfidenceStrategySelector",
    )
    module_names = (
        "intergrax.runtime.self_healing.knowledge_evolution.governance.service",
        "intergrax.runtime.self_healing.knowledge_evolution.governance.default_policy",
        "intergrax.runtime.self_healing.knowledge_evolution.governance.basic_integrity_validator",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden_tokens:
            assert token.lower() not in lowered


def test_integrity_validator_contract_is_swappable() -> None:
    @dataclass(frozen=True, slots=True)
    class AlwaysValidValidator:
        def validate_scope(self, **kwargs: object) -> object:
            from intergrax.contracts.self_healing.knowledge_evolution.governance.integrity import (
                StrategyKnowledgeIntegrityReport,
            )

            _ = kwargs
            return StrategyKnowledgeIntegrityReport(
                tenant_id=_TENANT,
                strategy_id=_STRATEGY,
                context_fingerprint=_CONTEXT_FP,
                is_valid=True,
                issues=(),
            )

    validator = AlwaysValidValidator()
    assert isinstance(validator, StrategyKnowledgeIntegrityValidator)
