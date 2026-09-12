# © Artur Czarnecki. All rights reserved.

"""Resume and duplicate-cohort guards for R6-LIVE multi-model execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.checkpoint import (
    load_checkpoint,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)


class CohortResumeAction(StrEnum):
    PROCEED = "PROCEED"
    SKIP_DUPLICATE_FINALIZED = "SKIP_DUPLICATE_FINALIZED"
    BLOCK_RESUME_REQUIRED = "BLOCK_RESUME_REQUIRED"


@dataclass(frozen=True, slots=True)
class CohortResumeDecision:
    action: CohortResumeAction
    session_state: QualificationSessionState | None = None


def decide_cohort_resume(
    session_dir: Path,
    *,
    resume: bool,
    finalize_only: bool,
) -> CohortResumeDecision:
    """Fail-closed resume policy before mutating provider env or invoking the session runner."""
    checkpoint = load_checkpoint(session_dir)
    if checkpoint is None:
        return CohortResumeDecision(action=CohortResumeAction.PROCEED)
    state = checkpoint.state
    if state is QualificationSessionState.FINALIZED and not finalize_only:
        return CohortResumeDecision(
            action=CohortResumeAction.SKIP_DUPLICATE_FINALIZED,
            session_state=state,
        )
    if state is QualificationSessionState.CREATED:
        return CohortResumeDecision(action=CohortResumeAction.PROCEED, session_state=state)
    if resume or finalize_only:
        return CohortResumeDecision(action=CohortResumeAction.PROCEED, session_state=state)
    return CohortResumeDecision(
        action=CohortResumeAction.BLOCK_RESUME_REQUIRED,
        session_state=state,
    )


__all__ = [
    "CohortResumeAction",
    "CohortResumeDecision",
    "decide_cohort_resume",
]
