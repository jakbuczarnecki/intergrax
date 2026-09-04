# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Council → canonical Execution deliberation seam (DS-COUNCIL).

Hosts bounded multi-participant deliberation: parallel independent proposals,
structured disagreement, synthesis attempts, and typed deadlock outcomes.
Council does not finalize decisions, verify output, or import Nexus/providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_context_visibility import (
    DeliberationContextId,
    ParticipantContextVisibilityPolicy,
    is_context_visible,
)
from intergrax.contracts.decision_disagreement import DecisionDisagreementArtifact
from intergrax.contracts.decision_identity import DecisionIdentity, next_decision_version
from intergrax.contracts.decision_participants import (
    ParticipantBinding,
    ParticipantRoleDefinition,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifactKind,
    DecisionBranchId,
    DecisionProposalRef,
    candidate_decision,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.council_strategy import (
    CouncilDeadlockReasonCode,
    CouncilDeliberationInput,
    CouncilDeliberationResult,
    CouncilDisagreementAnalyzer,
    CouncilParticipantProposal,
    CouncilResolutionDisposition,
    CouncilRoundState,
    CouncilStrategy,
    CouncilSynthesisAttempt,
    CouncilSynthesisDisposition,
    CouncilSynthesizer,
    council_deliberation_result_deadlock,
    council_deliberation_result_synthesized,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.concurrent_execution_work import (
    execute_concurrent_execution_work,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

T = TypeVar("T")

_UNTRUSTED_PEER_PROPOSAL_PREFIX = "[untrusted-participant-proposal]"
_UNTRUSTED_DISAGREEMENT_PREFIX = "[untrusted-disagreement-context]"


@dataclass(frozen=True, slots=True)
class MaterializedParticipantDeliberationInput:
    """Auditable participant input after context visibility materialization."""

    participant_id: str
    role_id: str
    trusted_instruction: str
    messages: tuple[ChatMessage, ...]
    visible_context_ids: tuple[DeliberationContextId, ...]


def participant_proposal_branch_id(participant_id: str) -> DecisionBranchId:
    """Deterministic proposal branch identity for one participant."""
    return validate_decision_branch_id(f"proposal-{participant_id}")


def _visibility_policy_for_role(
    strategy: CouncilStrategy,
    role_id: str,
) -> ParticipantContextVisibilityPolicy | None:
    for policy in strategy.visibility.policies:
        if policy.role_id == role_id:
            return policy
    return None


def _surface_messages_for_role(
    *,
    deliberation_input: CouncilDeliberationInput[T],
    policy: ParticipantContextVisibilityPolicy,
) -> tuple[tuple[ChatMessage, ...], tuple[DeliberationContextId, ...]]:
    visible_messages: list[ChatMessage] = []
    visible_ids: list[DeliberationContextId] = []
    for surface in deliberation_input.context_surfaces:
        if is_context_visible(policy, surface.context_id):
            visible_messages.extend(surface.messages)
            visible_ids.append(surface.context_id)
    return tuple(visible_messages), tuple(visible_ids)


def materialize_participant_deliberation_input(
    *,
    strategy: CouncilStrategy,
    deliberation_input: CouncilDeliberationInput[T],
    binding: ParticipantBinding,
    role_definition: ParticipantRoleDefinition,
    disagreement: DecisionDisagreementArtifact | None = None,
) -> MaterializedParticipantDeliberationInput:
    """Materialize one participant inference input honoring visibility policy."""
    policy = _visibility_policy_for_role(strategy, binding.role_id)
    if policy is None:
        raise ValueError(
            f"no visibility policy for participant role: {binding.role_id!r}",
        )
    context_messages, visible_ids = _surface_messages_for_role(
        deliberation_input=deliberation_input,
        policy=policy,
    )
    trusted_instruction = role_definition.instruction
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=trusted_instruction),
    ]
    messages.extend(deliberation_input.task_messages)
    messages.extend(context_messages)
    if disagreement is not None:
        disagreement_summary = _format_disagreement_summary(disagreement)
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    f"{_UNTRUSTED_DISAGREEMENT_PREFIX}\n{disagreement_summary}"
                ),
            ),
        )
    return MaterializedParticipantDeliberationInput(
        participant_id=binding.participant_id,
        role_id=binding.role_id,
        trusted_instruction=trusted_instruction,
        messages=tuple(messages),
        visible_context_ids=visible_ids,
    )


def _format_disagreement_summary(disagreement: DecisionDisagreementArtifact) -> str:
    conflict_lines = [
        f"{conflict.dimension}: {conflict.summary}"
        for conflict in disagreement.conflicts
    ]
    question_lines = [question.question for question in disagreement.unresolved_questions]
    sections = ["conflicts:"] + conflict_lines
    if question_lines:
        sections.extend(["unresolved_questions:"] + question_lines)
    return "\n".join(sections)


def participant_inference_execution_request(
    *,
    materialized: MaterializedParticipantDeliberationInput,
    deliberation_input: CouncilDeliberationInput[T],
    binding: ParticipantBinding,
) -> ExecutionRequest[tuple[ChatMessage, ...], T]:
    """Build canonical inference ExecutionRequest for one Council participant."""
    return ExecutionRequest(
        input=materialized.messages,
        output_type=deliberation_input.output_type,
        inference_profile_id=binding.inference_profile_id,
    )


def _role_definition_for_binding(
    strategy: CouncilStrategy,
    binding: ParticipantBinding,
) -> ParticipantRoleDefinition:
    for role in strategy.participants.roles:
        if role.role_id == binding.role_id:
            return role
    raise ValueError(f"unknown role for participant binding: {binding.role_id!r}")


def _proposal_refs(
    proposals: tuple[CouncilParticipantProposal[T], ...],
) -> tuple[DecisionProposalRef, ...]:
    return tuple(proposal.proposal_ref for proposal in proposals)


def _build_participant_proposal(
    *,
    deliberation_input: CouncilDeliberationInput[T],
    binding: ParticipantBinding,
    payload: T,
    rationale_summary: str,
) -> CouncilParticipantProposal[T]:
    branch_id = participant_proposal_branch_id(binding.participant_id)
    lineage = decision_version_lineage(
        current=decision_lineage_ref(deliberation_input.identity.version, branch_id),
    )
    candidate = candidate_decision(
        identity=deliberation_input.identity,
        artifact_kind=deliberation_input.artifact_kind,
        payload=payload,
        lineage=lineage,
    )
    proposal_ref = candidate_decision_ref(candidate)
    return CouncilParticipantProposal(
        participant_id=binding.participant_id,
        role_id=binding.role_id,
        inference_profile_id=binding.inference_profile_id,
        proposal_ref=proposal_ref,
        candidate=candidate,
        rationale_summary=rationale_summary,
    )


def _execution_budget_allows_work(ledger: ExecutionBudgetLedger | None) -> bool:
    if ledger is None:
        return True
    available = ledger.snapshot_root_available()
    limits = (
        available.max_input_tokens,
        available.max_output_tokens,
        available.max_total_tokens,
        available.max_llm_calls,
        available.max_tool_calls,
        available.max_rag_invocations,
        available.max_websearch_invocations,
        available.max_planner_iterations,
        available.max_replans,
    )
    for limit in limits:
        if limit is not None and limit <= 0:
            return False
    max_wall_time_seconds = available.max_wall_time_seconds
    if max_wall_time_seconds is not None and max_wall_time_seconds <= 0:
        return False
    return True


async def execute_parallel_participant_proposals(
    *,
    strategy: CouncilStrategy,
    deliberation_input: CouncilDeliberationInput[T],
    work_port: ExecutionWorkPort[tuple[ChatMessage, ...], T, T],
    disagreement: DecisionDisagreementArtifact | None = None,
) -> tuple[CouncilParticipantProposal[T], ...]:
    """Produce independent participant proposals concurrently via Execution work."""
    bindings = strategy.participants.participants
    materialized_inputs: list[MaterializedParticipantDeliberationInput] = []
    requests: list[ExecutionRequest[tuple[ChatMessage, ...], T]] = []
    for binding in bindings:
        role_definition = _role_definition_for_binding(strategy, binding)
        materialized = materialize_participant_deliberation_input(
            strategy=strategy,
            deliberation_input=deliberation_input,
            binding=binding,
            role_definition=role_definition,
            disagreement=disagreement,
        )
        materialized_inputs.append(materialized)
        requests.append(
            participant_inference_execution_request(
                materialized=materialized,
                deliberation_input=deliberation_input,
                binding=binding,
            ),
        )
    outputs = await execute_concurrent_execution_work(
        work_port,
        tuple(requests),
    )
    proposals: list[CouncilParticipantProposal[T]] = []
    for binding, payload in zip(bindings, outputs, strict=True):
        proposals.append(
            _build_participant_proposal(
                deliberation_input=deliberation_input,
                binding=binding,
                payload=payload,
                rationale_summary="participant proposal produced",
            ),
        )
    return tuple(proposals)


async def execute_parallel_participant_proposals_resilient(
    *,
    strategy: CouncilStrategy,
    deliberation_input: CouncilDeliberationInput[T],
    work_port: ExecutionWorkPort[tuple[ChatMessage, ...], T, T],
    disagreement: DecisionDisagreementArtifact | None = None,
) -> tuple[CouncilParticipantProposal[T], ...]:
    """Produce proposals concurrently; omit failed participants instead of aborting."""
    bindings = strategy.participants.participants
    materialized_inputs: list[MaterializedParticipantDeliberationInput] = []
    requests: list[ExecutionRequest[tuple[ChatMessage, ...], T]] = []
    for binding in bindings:
        role_definition = _role_definition_for_binding(strategy, binding)
        materialized = materialize_participant_deliberation_input(
            strategy=strategy,
            deliberation_input=deliberation_input,
            binding=binding,
            role_definition=role_definition,
            disagreement=disagreement,
        )
        materialized_inputs.append(materialized)
        requests.append(
            participant_inference_execution_request(
                materialized=materialized,
                deliberation_input=deliberation_input,
                binding=binding,
            ),
        )
    proposals: list[CouncilParticipantProposal[T]] = []
    for binding, request in zip(bindings, requests, strict=True):
        try:
            payload = await work_port.execute(request)
        except Exception:
            continue
        proposals.append(
            _build_participant_proposal(
                deliberation_input=deliberation_input,
                binding=binding,
                payload=payload,
                rationale_summary="participant proposal produced",
            ),
        )
    return tuple(proposals)


def _same_decision_boundary(
    identity: DecisionIdentity,
    proposal_refs: tuple[DecisionProposalRef, ...],
) -> bool:
    for ref in proposal_refs:
        if ref.identity != identity:
            return False
    return True


async def execute_council_deliberation(
    *,
    strategy: CouncilStrategy,
    deliberation_input: CouncilDeliberationInput[T],
    work_port: ExecutionWorkPort[tuple[ChatMessage, ...], T, T],
    disagreement_analyzer: CouncilDisagreementAnalyzer[T],
    synthesizer: CouncilSynthesizer[T],
    budget_ledger: ExecutionBudgetLedger | None = None,
    resilient_participant_failures: bool = False,
) -> CouncilDeliberationResult[T]:
    """Run bounded Council deliberation and return one typed semantic result."""
    max_rounds = strategy.round_policy.max_rounds
    minimum_successful = strategy.synthesis.failure_policy.minimum_successful_participants
    disagreement: DecisionDisagreementArtifact | None = None
    proposals: tuple[CouncilParticipantProposal[T], ...] = ()
    round_number = 0

    for round_index in range(max_rounds):
        round_number = round_index + 1
        if not _execution_budget_allows_work(budget_ledger):
            return council_deliberation_result_deadlock(
                result_payload_type=deliberation_input.output_type,
                proposal_refs=_proposal_refs(proposals),
                disagreement=disagreement,
                rounds_used=round_number if round_number > 0 else 1,
                deadlock_reason=CouncilDeadlockReasonCode.EXECUTION_BUDGET_EXHAUSTED,
            )
        if resilient_participant_failures:
            proposals = await execute_parallel_participant_proposals_resilient(
                strategy=strategy,
                deliberation_input=deliberation_input,
                work_port=work_port,
                disagreement=disagreement if round_number > 1 else None,
            )
        else:
            proposals = await execute_parallel_participant_proposals(
                strategy=strategy,
                deliberation_input=deliberation_input,
                work_port=work_port,
                disagreement=disagreement if round_number > 1 else None,
            )
        if len(proposals) < minimum_successful:
            reason = (
                CouncilDeadlockReasonCode.INSUFFICIENT_PROPOSALS
                if len(proposals) < 2
                else CouncilDeadlockReasonCode.PARTICIPANT_FAILURE
            )
            return council_deliberation_result_deadlock(
                result_payload_type=deliberation_input.output_type,
                proposal_refs=_proposal_refs(proposals),
                disagreement=disagreement,
                rounds_used=round_number,
                deadlock_reason=reason,
            )
        disagreement = disagreement_analyzer.analyze(proposals=proposals)
        if not _same_decision_boundary(deliberation_input.identity, disagreement.proposal_refs):
            raise ValueError("disagreement proposal refs must share decision boundary")
        round_state = CouncilRoundState(
            round_number=round_number,
            proposal_refs=_proposal_refs(proposals),
            disagreement=disagreement,
        )
        synthesis_attempt = synthesizer.synthesize(
            proposals=proposals,
            disagreement=disagreement,
            round_state=round_state,
            synthesis_instruction=strategy.synthesis.synthesis_instruction,
        )
        if synthesis_attempt.disposition == CouncilSynthesisDisposition.RESOLVED:
            if synthesis_attempt.candidate is None:
                raise ValueError("resolved synthesis attempt requires candidate")
            return council_deliberation_result_synthesized(
                candidate=synthesis_attempt.candidate,
                proposal_refs=_proposal_refs(proposals),
                disagreement=disagreement,
                rounds_used=round_number,
            )
        if round_number >= max_rounds:
            return council_deliberation_result_deadlock(
                result_payload_type=deliberation_input.output_type,
                proposal_refs=_proposal_refs(proposals),
                disagreement=disagreement,
                rounds_used=round_number,
                deadlock_reason=CouncilDeadlockReasonCode.PERSISTENT_DISAGREEMENT,
            )

    raise ValueError("council deliberation completed without bounded outcome")


def synthesis_candidate_from_proposals(
    *,
    identity: DecisionIdentity,
    artifact_kind: DecisionArtifactKind,
    payload: T,
    parent_proposals: tuple[CouncilParticipantProposal[T], ...],
    branch_id: str = "synthesis",
) -> CandidateDecision[T]:
    """Build synthesis CandidateDecision with lineage parents from source proposals."""
    validate_decision_artifact_kind(artifact_kind)
    synthesis_version = next_decision_version(identity.version)
    synthesis_identity = DecisionIdentity(
        decision_id=identity.decision_id,
        version=synthesis_version,
        scope=identity.scope,
        tenant_id=identity.tenant_id,
        execution=identity.execution,
    )
    parents = tuple(
        proposal.candidate.lineage.current for proposal in parent_proposals
    )
    lineage = decision_version_lineage(
        current=decision_lineage_ref(
            synthesis_version,
            validate_decision_branch_id(branch_id),
        ),
        parents=parents,
    )
    return candidate_decision(
        identity=synthesis_identity,
        artifact_kind=artifact_kind,
        payload=payload,
        lineage=lineage,
    )


def untrusted_proposal_message_content(proposal_summary: str) -> str:
    """Format participant proposal content as untrusted deliberation data."""
    return f"{_UNTRUSTED_PEER_PROPOSAL_PREFIX}\n{proposal_summary}"
