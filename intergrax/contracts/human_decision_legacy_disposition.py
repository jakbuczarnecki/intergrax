# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MP-4R6 — legacy human decision data disposition contracts (audit/migration only).

These types are not decision authority, HITL authority, or runtime execution inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypedDict

from intergrax.contracts.human_approver import HumanApproverEvidence


class HumanDecisionLegacyDispositionStrategy(str, Enum):
    """Offline disposition policy selection — not semantic authority."""

    PROVENANCE_RECOVERY = "provenance_recovery"
    HISTORY_ONLY_QUARANTINE = "history_only_quarantine"
    CONTROLLED_DELETE = "controlled_delete"
    NO_DATA_PRESENT = "no_data_present"


class LegacyHumanDecisionProvenanceStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class LegacyHumanDecisionArchiveRecord:
    """
    Read-only historical view of a persisted human decision row.

    Explicitly excludes ``HumanApproverEvidence`` — not usable for authorization.
    """

    decision_id: str
    tenant_id: str
    task_id: str
    user_id: str
    human_request_id: str
    verdict: str
    response_text: str
    escalation_level: int
    escalation_target: str | None
    agent_id: str | None
    run_id: str | None
    notes: str
    created_at_utc: str
    provenance_status: LegacyHumanDecisionProvenanceStatus

    def to_archive_payload(self) -> LegacyHumanDecisionArchivePayload:
        """Deterministic, non-authoritative archive JSON object — explicit field projection."""
        return {
            "decision_id": self.decision_id,
            "tenant_id": self.tenant_id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "human_request_id": self.human_request_id,
            "verdict": self.verdict,
            "response_text": self.response_text,
            "escalation_level": self.escalation_level,
            "escalation_target": self.escalation_target,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "notes": self.notes,
            "created_at_utc": self.created_at_utc,
            "provenance_status": self.provenance_status.value,
        }


class LegacyHumanDecisionArchivePayload(TypedDict):
    """Stable archive JSON shape — excludes approver authority fields."""

    decision_id: str
    tenant_id: str
    task_id: str
    user_id: str
    human_request_id: str
    verdict: str
    response_text: str
    escalation_level: int
    escalation_target: str | None
    agent_id: str | None
    run_id: str | None
    notes: str
    created_at_utc: str
    provenance_status: str


def serialize_legacy_human_decision_archive_record(
    record: LegacyHumanDecisionArchiveRecord,
) -> LegacyHumanDecisionArchivePayload:
    """Contract-level archive serializer — no reflection."""
    return record.to_archive_payload()


@dataclass(frozen=True, slots=True)
class HumanDecisionLegacyDataAssessment:
    """Neutral inventory counts for a single SQLite human-decisions database."""

    rows_total: int
    rows_with_approver_json: int
    rows_missing_approver_json: int
    rows_malformed_approver_json: int
    rows_recoverable: int
    rows_unrecoverable: int


@dataclass(frozen=True, slots=True)
class HumanDecisionLegacyDispositionReport:
    """Result of an offline disposition pass (dry-run or applied recovery)."""

    assessment: HumanDecisionLegacyDataAssessment
    dry_run: bool
    rows_scanned: int
    rows_valid: int
    rows_missing_provenance: int
    rows_malformed_provenance: int
    rows_recoverable: int
    rows_unrecoverable: int
    rows_recovered: int
    rows_unchanged: int
    rows_deleted: int
    rows_quarantined: int


class HumanDecisionApproverRecoverySource(Protocol):
    """External exact provenance for legacy rows — never inferred from ``user_id``."""

    def recover_approver(
        self,
        *,
        decision_id: str,
        tenant_id: str,
    ) -> HumanApproverEvidence | None:
        """Return exact canonical approver evidence or ``None`` when unavailable."""
