# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge integrity checks — structural, not cryptographic (SELF-HEALING R5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_repository import (
    StrategyKnowledgeAuditRepository,
)
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.repository import StrategyKnowledgeRepository


class StrategyKnowledgeIntegrityIssueCode(StrEnum):
    VERSION_GAP = "VERSION_GAP"
    SUPERSEDES_MISMATCH = "SUPERSEDES_MISMATCH"
    MISSING_AUDIT_CHANGE_RECORD = "MISSING_AUDIT_CHANGE_RECORD"
    AUDIT_VERSION_MISMATCH = "AUDIT_VERSION_MISMATCH"
    ORPHAN_AUDIT_RECORD = "ORPHAN_AUDIT_RECORD"


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeIntegrityIssue:
    code: StrategyKnowledgeIntegrityIssueCode
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("message required")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeIntegrityReport:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    is_valid: bool
    issues: tuple[StrategyKnowledgeIntegrityIssue, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if self.is_valid and self.issues:
            raise ValueError("is_valid cannot be true when issues are present")
        if not self.is_valid and not self.issues:
            raise ValueError("invalid report must include issues")


@runtime_checkable
class StrategyKnowledgeIntegrityValidator(Protocol):
    def validate_scope(
        self,
        *,
        knowledge_repository: StrategyKnowledgeRepository,
        audit_repository: StrategyKnowledgeAuditRepository,
        scope: StrategyKnowledgeProfileQuery,
    ) -> StrategyKnowledgeIntegrityReport: ...


__all__ = [
    "StrategyKnowledgeIntegrityIssue",
    "StrategyKnowledgeIntegrityIssueCode",
    "StrategyKnowledgeIntegrityReport",
    "StrategyKnowledgeIntegrityValidator",
]
