# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MP-4R6 — legacy human decision data disposition contracts (audit/migration only).

These types are not decision authority, HITL authority, or runtime execution inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.human_approver import HumanApproverEvidence


class HumanDecisionLegacyDispositionStrategy(str, Enum):
    """Offline disposition policy selection — not semantic authority."""

    PROVENANCE_RECOVERY = "provenance_recovery"
    HISTORY_ONLY_QUARANTINE = "history_only_quarantine"
    CONTROLLED_ARCHIVE = "controlled_archive"
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
