# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Structural integrity checks for knowledge and audit alignment (R5.6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_query import (
    StrategyKnowledgeChangeRecordQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_repository import (
    StrategyKnowledgeAuditRepository,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.integrity import (
    StrategyKnowledgeIntegrityIssue,
    StrategyKnowledgeIntegrityIssueCode,
    StrategyKnowledgeIntegrityReport,
)
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
    StrategyKnowledgeRevisionQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.repository import StrategyKnowledgeRepository


@dataclass(frozen=True, slots=True)
class BasicStrategyKnowledgeIntegrityValidator:
    validator_id: str = "platform.basic_knowledge_integrity"

    def validate_scope(
        self,
        *,
        knowledge_repository: StrategyKnowledgeRepository,
        audit_repository: StrategyKnowledgeAuditRepository,
        scope: StrategyKnowledgeProfileQuery,
    ) -> StrategyKnowledgeIntegrityReport:
        issues: list[StrategyKnowledgeIntegrityIssue] = []
        revisions = knowledge_repository.list_revisions(
            StrategyKnowledgeRevisionQuery(
                tenant_id=scope.tenant_id,
                strategy_id=scope.strategy_id,
                context_fingerprint=scope.context_fingerprint,
                limit=10_000,
            ),
        )
        change_records = audit_repository.list_change_records(
            StrategyKnowledgeChangeRecordQuery(
                tenant_id=scope.tenant_id,
                strategy_id=scope.strategy_id,
                context_fingerprint=scope.context_fingerprint,
                limit=10_000,
            ),
        )
        revision_by_id = {row.revision_id: row for row in revisions}
        records_by_revision = {row.revision_id: row for row in change_records}

        expected_version = 1
        for revision in revisions:
            profile = revision.profile
            if profile.knowledge_version != expected_version:
                issues.append(
                    StrategyKnowledgeIntegrityIssue(
                        code=StrategyKnowledgeIntegrityIssueCode.VERSION_GAP,
                        message=(
                            f"Expected knowledge version {expected_version}, "
                            f"found {profile.knowledge_version} in revision {revision.revision_id}."
                        ),
                    ),
                )
            if revision.previous_knowledge_version is not None:
                if profile.supersedes_version != revision.previous_knowledge_version:
                    issues.append(
                        StrategyKnowledgeIntegrityIssue(
                            code=StrategyKnowledgeIntegrityIssueCode.SUPERSEDES_MISMATCH,
                            message=f"Revision {revision.revision_id} supersedes metadata inconsistent.",
                        ),
                    )
            audit_row = records_by_revision.get(revision.revision_id)
            if audit_row is None:
                issues.append(
                    StrategyKnowledgeIntegrityIssue(
                        code=StrategyKnowledgeIntegrityIssueCode.MISSING_AUDIT_CHANGE_RECORD,
                        message=f"No audit change record for revision {revision.revision_id}.",
                    ),
                )
            elif audit_row.new_knowledge_version != profile.knowledge_version:
                issues.append(
                    StrategyKnowledgeIntegrityIssue(
                        code=StrategyKnowledgeIntegrityIssueCode.AUDIT_VERSION_MISMATCH,
                        message=(
                            f"Audit record {audit_row.change_id} version "
                            f"{audit_row.new_knowledge_version} != profile {profile.knowledge_version}."
                        ),
                    ),
                )
            expected_version = profile.knowledge_version + 1

        for record in change_records:
            if record.revision_id not in revision_by_id:
                issues.append(
                    StrategyKnowledgeIntegrityIssue(
                        code=StrategyKnowledgeIntegrityIssueCode.ORPHAN_AUDIT_RECORD,
                        message=f"Audit change {record.change_id} references missing revision {record.revision_id}.",
                    ),
                )

        return StrategyKnowledgeIntegrityReport(
            tenant_id=scope.tenant_id,
            strategy_id=scope.strategy_id,
            context_fingerprint=scope.context_fingerprint,
            is_valid=not issues,
            issues=tuple(issues),
        )


__all__ = ["BasicStrategyKnowledgeIntegrityValidator"]
