# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision revision lifecycle helpers (DS-REV-01).

Mint revised candidates and transition lifecycle stages without performing
revision work, model calls, or execution side effects.
"""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.decision_identity import (
    DecisionIdentity,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifactKind,
    DecisionVersionLineage,
    candidate_decision,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionAuthorization,
    DecisionRevisionDecision,
    DecisionRevisionDisposition,
    DecisionRevisionState,
    proposal_refs_match,
)
from intergrax.contracts.decision_verification import VerificationResult

T = TypeVar("T")


class DecisionRevisionAuthorizationMismatchError(ValueError):
    """Raised when revision authorization does not match the target proposal."""


def validate_revision_authorization_for_candidate(
    *,
    authorization: DecisionRevisionAuthorization,
    candidate: CandidateDecision[T],
) -> None:
    """Reject stale or sibling-branch reuse of one revision authorization."""
    if type(authorization) is not DecisionRevisionAuthorization:
        raise TypeError("authorization must be DecisionRevisionAuthorization")
    if type(candidate) is not CandidateDecision:
        raise TypeError("candidate must be CandidateDecision")
    candidate_ref = candidate_decision_ref(candidate)
    if not proposal_refs_match(authorization.proposal_ref, candidate_ref):
        raise DecisionRevisionAuthorizationMismatchError(
            "revision authorization proposal_ref must match candidate proposal_ref",
        )


def mint_revised_candidate_decision(
    *,
    challenged: CandidateDecision[T],
    authorization: DecisionRevisionAuthorization,
    artifact_kind: DecisionArtifactKind | str,
    revised_payload: T,
    revision_state: DecisionRevisionState,
) -> tuple[CandidateDecision[T], DecisionRevisionState]:
    """Mint one immutable revised candidate with exact parent lineage binding."""
    if type(challenged) is not CandidateDecision:
        raise TypeError("challenged must be CandidateDecision")
    if type(authorization) is not DecisionRevisionAuthorization:
        raise TypeError("authorization must be DecisionRevisionAuthorization")
    if type(revision_state) is not DecisionRevisionState:
        raise TypeError("revision_state must be DecisionRevisionState")
    validate_revision_authorization_for_candidate(
        authorization=authorization,
        candidate=challenged,
    )
    challenged_ref = candidate_decision_ref(challenged)
    if not proposal_refs_match(revision_state.proposal_ref, challenged_ref):
        raise DecisionRevisionAuthorizationMismatchError(
            "revision_state proposal_ref must match challenged candidate proposal_ref",
        )
    expected_revision_number = revision_state.revision_count + 1
    if authorization.revision_number != expected_revision_number:
        raise DecisionRevisionAuthorizationMismatchError(
            "revision authorization revision_number must equal revision_count + 1",
        )
    new_version = next_decision_version(challenged.identity.version)
    new_identity = DecisionIdentity(
        decision_id=challenged.identity.decision_id,
        version=new_version,
        scope=challenged.identity.scope,
        tenant_id=challenged.identity.tenant_id,
        execution=challenged.identity.execution,
    )
    new_lineage = decision_version_lineage(
        current=decision_lineage_ref(
            new_version,
            challenged.lineage.current.branch_id,
        ),
        parents=(challenged_ref.lineage_ref,),
    )
    resolved_kind = (
        artifact_kind
        if type(artifact_kind) is DecisionArtifactKind
        else validate_decision_artifact_kind(artifact_kind)
    )
    revised = candidate_decision(
        identity=new_identity,
        artifact_kind=resolved_kind,
        payload=revised_payload,
        lineage=new_lineage,
    )
    next_state = DecisionRevisionState(
        proposal_ref=candidate_decision_ref(revised),
        revision_count=revision_state.revision_count + 1,
    )
    return revised, next_state


def transition_lifecycle_for_revision(
    *,
    lifecycle_state: DecisionLifecycleState,
    verification_result: VerificationResult,
    revision_decision: DecisionRevisionDecision,
) -> DecisionLifecycleState:
    """Apply canonical VERIFICATION → REVISION transition when revision is allowed."""
    if type(lifecycle_state) is not DecisionLifecycleState:
        raise TypeError("lifecycle_state must be DecisionLifecycleState")
    if type(verification_result) is not VerificationResult:
        raise TypeError("verification_result must be VerificationResult")
    if type(revision_decision) is not DecisionRevisionDecision:
        raise TypeError("revision_decision must be DecisionRevisionDecision")
    if lifecycle_state.stage is not DecisionLifecycleStage.VERIFICATION:
        raise ValueError(
            "revision lifecycle transition requires current stage verification",
        )
    if not proposal_refs_match(
        revision_decision.proposal_ref,
        verification_result.proposal_ref,
    ):
        raise ValueError(
            "revision decision proposal_ref must match verification_result.proposal_ref",
        )
    if revision_decision.disposition is not DecisionRevisionDisposition.ALLOWED:
        raise ValueError(
            "revision lifecycle transition requires DecisionRevisionDisposition.ALLOWED",
        )
    if lifecycle_state.identity != verification_result.proposal_ref.identity:
        raise ValueError(
            "lifecycle identity must match verification_result proposal identity",
        )
    return transition_decision_lifecycle(
        lifecycle_state,
        DecisionLifecycleStage.REVISION,
    )
