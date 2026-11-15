# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authority-negative validation for scheduler transport metadata (SX-F13)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.contracts.task_metadata_keys import TaskMetadataKey

__all__ = [
    "RECURRENCE_METADATA_KEYS",
    "RECURRENCE_OCCURRENCE_KEY",
    "RECURRENCE_RULE_KEY",
    "ScheduledResumeMetadataValidationError",
    "has_recurrence_metadata",
    "reject_recurrence_metadata",
    "validate_scheduled_resume_metadata",
    "scheduled_resume_forbidden_metadata_keys",
]

#: Keys that mint future executions; only ``RecurrenceMaterializer`` rows may carry them.
RECURRENCE_RULE_KEY = "recurrence_rule"
RECURRENCE_OCCURRENCE_KEY = "recurrence_occurrence_utc"
RECURRENCE_METADATA_KEYS: frozenset[str] = frozenset({RECURRENCE_RULE_KEY, RECURRENCE_OCCURRENCE_KEY})


class ScheduledResumeMetadataValidationError(ValueError):
    """Resume metadata carries authority-bearing or identity-overriding semantics."""


_EXACT_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        TaskMetadataKey.HUMAN_APPROVED,
        TaskMetadataKey.HUMAN_REJECTED,
        TaskMetadataKey.HUMAN_ESCALATED,
        TaskMetadataKey.HUMAN_RESPONSE,
        TaskMetadataKey.HUMAN_DECISION,
        TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST,
        TaskMetadataKey.GOVERNANCE_INTERRUPT,
        TaskMetadataKey.GOVERNANCE_PAUSE,
        TaskMetadataKey.GOVERNANCE_PAUSE_RECORD,
        TaskMetadataKey.REQUIRE_HUMAN_APPROVAL,
        TaskMetadataKey.REQUIRE_HUMAN_ON_CRITICAL,
        TaskMetadataKey.HIGH_RISK,
        TaskMetadataKey.RESUME_TOKEN,
        TaskMetadataKey.CHECKPOINT_ID,
        "verdict",
        "response_text",
        "approver",
        "run_id",
        "attempt_id",
        "execution_id",
        "task_id",
        "tenant_id",
    }
)


def scheduled_resume_forbidden_metadata_keys() -> frozenset[str]:
    """Closed-world exact keys rejected at scheduler metadata boundaries."""
    return _EXACT_FORBIDDEN_KEYS


def _is_forbidden_metadata_key(key: str) -> bool:
    if key in _EXACT_FORBIDDEN_KEYS:
        return True
    lowered = key.lower()
    if lowered.startswith("governance_"):
        return True
    if lowered.startswith("approval") or lowered.startswith("authorization"):
        return True
    return False


def validate_scheduled_resume_metadata(metadata: Mapping[str, Any] | None) -> None:
    """Reject metadata that can mint Human/Governance authority or override identity."""
    if not metadata:
        return
    for key in metadata:
        if _is_forbidden_metadata_key(key):
            raise ScheduledResumeMetadataValidationError(
                f"Scheduled resume metadata key {key!r} is authority-bearing or "
                "identity-overriding and is rejected fail-closed.",
            )


def has_recurrence_metadata(metadata: Mapping[str, Any] | None) -> bool:
    """True when the metadata carries any recurrence-minting key."""
    return bool(metadata) and any(key in RECURRENCE_METADATA_KEYS for key in metadata)


def reject_recurrence_metadata(metadata: Mapping[str, Any] | None) -> None:
    """Public entry points must not start a recurrence chain through plain metadata (F-D7-11)."""
    if has_recurrence_metadata(metadata):
        raise ScheduledResumeMetadataValidationError(
            "Scheduled resume metadata carries recurrence keys; recurrence chains are started "
            "only through RecurrenceMaterializer.",
        )
