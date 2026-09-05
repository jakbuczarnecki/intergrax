# © Artur Czarnecki. All rights reserved.

"""Council helpers for DS-E2E qualification."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.council_strategy import (
    CouncilDeliberationInput,
    CouncilStrategy,
    CouncilSynthesisConfiguration,
    council_context_surface,
    council_participant_failure_policy,
    council_round_policy,
    council_strategy_kind,
)
from intergrax.contracts.decision_context_visibility import (
    DeliberationContextId,
    participant_context_visibility_configuration,
    participant_context_visibility_policy,
)
from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_participants import (
    participant_binding,
    participant_configuration,
    participant_role_definition,
)
from intergrax.contracts.decision_record import validate_decision_artifact_kind
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.contracts.council_strategy import (
    CouncilDeliberationResult,
    CouncilParticipantProposal,
    CouncilSynthesisAttempt,
    CouncilSynthesisDisposition,
    CouncilDisagreementAnalyzer,
    CouncilSynthesizer,
)
from intergrax.contracts.decision_disagreement import (
    DecisionDisagreementArtifact,
    decision_disagreement_artifact,
    disagreement_conflict,
    disagreement_position,
)
from intergrax.runtime.execution.council_deliberation import (
    execute_council_deliberation,
    synthesis_candidate_from_proposals,
)

from testing_support.decision_e2e.payloads import QualificationRecommendation


class QualificationDisagreementAnalyzer(
    CouncilDisagreementAnalyzer[QualificationRecommendation],
):
    def analyze(
        self,
        *,
        proposals: tuple[
            CouncilParticipantProposal[QualificationRecommendation],
            ...,
        ],
    ) -> DecisionDisagreementArtifact:
        proposal_refs = tuple(proposal.proposal_ref for proposal in proposals)
        positions = tuple(
            disagreement_position(
                proposal_ref=proposal.proposal_ref,
                summary=proposal.candidate.artifact.content.recommendation.strip(),
            )
            for proposal in proposals
        )
        conflicts = (
            disagreement_conflict(
                dimension="recommendation",
                proposal_refs=proposal_refs,
                summary="Participants produced distinct recommendations.",
            ),
        )
        return decision_disagreement_artifact(
            proposal_refs=proposal_refs,
            positions=positions,
            conflicts=conflicts,
        )


@dataclass(slots=True)
class QualificationCouncilSynthesizer(
    CouncilSynthesizer[QualificationRecommendation],
):
    disagreement_context_seen: bool = False

    def synthesize(
        self,
        *,
        proposals: tuple[
            CouncilParticipantProposal[QualificationRecommendation],
            ...,
        ],
        disagreement: DecisionDisagreementArtifact,
        round_state: object,
        synthesis_instruction: str,
    ) -> CouncilSynthesisAttempt[QualificationRecommendation]:
        del round_state, synthesis_instruction
        self.disagreement_context_seen = disagreement is not None
        chosen = proposals[0].candidate.artifact.content
        candidate = synthesis_candidate_from_proposals(
            identity=proposals[0].candidate.identity,
            artifact_kind=proposals[0].candidate.artifact.kind,
            payload=QualificationRecommendation(
                recommendation=chosen.recommendation,
                confidence=chosen.confidence,
                rationale_summary="council synthesis from participant proposals",
            ),
            parent_proposals=proposals,
        )
        return CouncilSynthesisAttempt(
            disposition=CouncilSynthesisDisposition.RESOLVED,
            candidate=candidate,
        )


async def run_council_deliberation(
    composition,
    *,
    strategy,
    deliberation_input,
) -> tuple[CouncilDeliberationResult[QualificationRecommendation], int]:
    analyzer = QualificationDisagreementAnalyzer()
    synthesizer = QualificationCouncilSynthesizer()
    calls_before = composition.work_port.invocation_count
    result = await execute_council_deliberation(
        strategy=strategy,
        deliberation_input=deliberation_input,
        work_port=composition.work_port,
        disagreement_analyzer=analyzer,
        synthesizer=synthesizer,
        resilient_participant_failures=True,
    )
    invocations = composition.work_port.invocation_count - calls_before
    return result, invocations


def three_participant_strategy() -> CouncilStrategy:
    roles = (
        participant_role_definition(role_id="architect", instruction="Architect role."),
        participant_role_definition(role_id="risk", instruction="Risk role."),
        participant_role_definition(role_id="domain", instruction="Domain role."),
    )
    participants = participant_configuration(
        roles=roles,
        participants=(
            participant_binding(
                participant_id="participant-a",
                role_id="architect",
                inference_profile_id="profile-b",
            ),
            participant_binding(
                participant_id="participant-b",
                role_id="risk",
                inference_profile_id="profile-producer",
            ),
            participant_binding(
                participant_id="participant-c",
                role_id="domain",
                inference_profile_id="profile-c",
            ),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=participants,
        policies=tuple(
            participant_context_visibility_policy(
                role_id=role.role_id,
                visible_contexts=(DeliberationContextId("shared_context"),),
            )
            for role in roles
        ),
    )
    return CouncilStrategy(
        kind=council_strategy_kind(),
        participants=participants,
        visibility=visibility,
        round_policy=council_round_policy(max_rounds=1),
        synthesis=CouncilSynthesisConfiguration(
            synthesis_instruction="Synthesize one bounded recommendation.",
            failure_policy=council_participant_failure_policy(
                minimum_successful_participants=2,
            ),
        ),
    )


def council_deliberation_input(
    identity: DecisionIdentity,
    *,
    task_message: str,
) -> CouncilDeliberationInput[QualificationRecommendation]:
    return CouncilDeliberationInput(
        identity=identity,
        task_messages=(ChatMessage(role="user", content=task_message),),
        context_surfaces=(
            council_context_surface(
                context_id="shared_context",
                messages=(ChatMessage(role="user", content="Shared trusted context."),),
            ),
        ),
        output_type=QualificationRecommendation,
        artifact_kind=validate_decision_artifact_kind("decision_e2e_qualification"),
    )


async def run_with_execution_bindings(composition, identity: DecisionIdentity, coroutine):
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    identity_token = bind_active_execution_identity(
        run_id=identity.execution.run_id,
        attempt_id=identity.execution.attempt_id,
        execution_id=identity.execution.execution_id,
    )
    lifecycle_token = bind_active_decision_lifecycle_host(
        composition.lifecycle_for_identity(identity)[0],
    )
    try:
        return await coroutine()
    finally:
        reset_active_decision_lifecycle_host(lifecycle_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)
