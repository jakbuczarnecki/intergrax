# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authority-negative validation for scheduler transport metadata (SX-F13)."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.contracts.task_metadata_keys import TaskMetadataKey

__all__ = [
    "ScheduledResumeMetadataValidationError",
    "validate_scheduled_resume_metadata",
    "scheduled_resume_forbidden_metadata_keys",
]


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
