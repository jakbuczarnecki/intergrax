# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge governance orchestration — audit and validation only (R5.6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.events import StrategyKnowledgeUpdated
from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_repository import (
    StrategyKnowledgeAuditRepository,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.integrity import (
    StrategyKnowledgeIntegrityReport,
    StrategyKnowledgeIntegrityValidator,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.policy import (
    StrategyKnowledgeGovernanceAssessment,
    StrategyKnowledgeGovernancePolicy,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision
from intergrax.contracts.self_healing.knowledge_evolution.query import StrategyKnowledgeProfileQuery
from intergrax.contracts.self_healing.knowledge_evolution.repository import StrategyKnowledgeRepository
from intergrax.runtime.self_healing.knowledge_evolution.governance.change_record_builder import (
    build_change_record_from_revision,
)


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeGovernanceRecordResult:
    change_record: StrategyKnowledgeChangeRecord
    governance_assessment: StrategyKnowledgeGovernanceAssessment
    knowledge_updated_event: StrategyKnowledgeUpdated
    integrity_report: StrategyKnowledgeIntegrityReport | None = None


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeGovernanceService:
    audit_repository: StrategyKnowledgeAuditRepository
    governance_policy: StrategyKnowledgeGovernancePolicy
    integrity_validator: StrategyKnowledgeIntegrityValidator | None = None

    def record_knowledge_evolution(
        self,
        revision: StrategyKnowledgeRevision,
        *,
        knowledge_repository: StrategyKnowledgeRepository,
    ) -> StrategyKnowledgeGovernanceRecordResult:
        change_record = build_change_record_from_revision(revision)
        assessment = self.governance_policy.assess_change(change_record, revision)
        stored_record = self.audit_repository.append_change_record(change_record)
        event = StrategyKnowledgeUpdated(
            tenant_id=revision.profile.tenant_id,
            strategy_id=revision.profile.strategy_id,
            context_fingerprint=revision.profile.context_fingerprint,
            revision_id=revision.revision_id,
            change_id=stored_record.change_id,
            previous_knowledge_version=revision.previous_knowledge_version,
            new_knowledge_version=revision.profile.knowledge_version,
            governance_policy_id=assessment.policy_id,
            evolution_mechanism_id=revision.evolution_mechanism_id,
            recorded_at=revision.recorded_at,
        )
        stored_event = self.audit_repository.append_knowledge_updated_event(event)
        integrity_report: StrategyKnowledgeIntegrityReport | None = None
        if self.integrity_validator is not None:
            integrity_report = self.integrity_validator.validate_scope(
                knowledge_repository=knowledge_repository,
                audit_repository=self.audit_repository,
                scope=StrategyKnowledgeProfileQuery(
                    tenant_id=revision.profile.tenant_id,
                    strategy_id=revision.profile.strategy_id,
                    context_fingerprint=revision.profile.context_fingerprint,
                ),
            )
        return StrategyKnowledgeGovernanceRecordResult(
            change_record=stored_record,
            governance_assessment=assessment,
            knowledge_updated_event=stored_event,
            integrity_report=integrity_report,
        )


__all__ = ["StrategyKnowledgeGovernanceRecordResult", "StrategyKnowledgeGovernanceService"]
