# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    governance_requires_human_review_reason,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import DecisionResolution
from intergrax.contracts.decision_revision import (
    DecisionRevisionDisposition,
    decision_revision_policy,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStage,
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionCriticAuthorityConflictError,
    DecisionFlowGateCapabilities,
    DecisionFlowGovernanceSpec,
    DecisionFlowHostAction,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowScope,
    decision_identity_from_seed,
    validate_decision_critic_authority_config,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATH = Path("intergrax/runtime/decision_flow.py")
_FORBIDDEN_FRAGMENTS = (
    "runtime.nexus",
    "runtime.critic",
    "CriticOrchestrator",
    "Any",
    "cast(",
    "type: ignore",
    "getattr",
    "hasattr",
    "dict[str, Any]",
)


@dataclass(frozen=True, slots=True)
class Payload:
    text: str


@dataclass(frozen=True, slots=True)
class PassedStage:
    kind: str
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(self, candidate):
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class ChallengedStage:
    kind: str
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(self, candidate):
        proposal_ref = candidate_decision_ref(candidate)
        finding = verification_finding(
            code=validate_verification_finding_code("verification.test.challenged"),
            message="challenged",
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=verification_challenge(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind(self.kind),
                requirement_code=validate_verification_requirement_code(
                    "verification.test.requirement",
                ),
                finding=finding,
            ),
        )


@dataclass(slots=True)
class RecordingHumanReviewPort:
    pending: DecisionHumanReviewPending | None = None

    def request_review(self, request):
        from intergrax.runtime.decision_human_review import request_decision_human_review

        self.pending = request_decision_human_review(request)
        return self.pending


@dataclass(frozen=True, slots=True)
class DenyGovernanceEvaluator:
    action: object
    policy_context: object

    def evaluate(self, *, evaluation_input):
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.DENY,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


@dataclass(frozen=True, slots=True)
class AllowGovernanceEvaluator:
    action: object
    policy_context: object

    def evaluate(self, *, evaluation_input):
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.ALLOW,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


@dataclass(frozen=True, slots=True)
class RequireHumanGovernanceEvaluator:
    action: object
    policy_context: object

    def evaluate(self, *, evaluation_input):
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.REQUIRE_HUMAN,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


@dataclass(slots=True)
class FailingHumanReviewPort:
    def request_review(self, request):
        raise RuntimeError("transport unavailable")


def _governance_spec(evaluator) -> DecisionFlowGovernanceSpec[Payload]:
    action = evaluator.action
    policy = evaluator.policy_context
    return DecisionFlowGovernanceSpec(
        action=action,
        policy_context=policy,
        evaluator=evaluator,
    )


def _pipeline(stage: VerificationStage[Payload]) -> VerificationPipeline[Payload]:
    return VerificationPipeline(
        registry=verification_stage_registry(
            (
                VerificationStageRegistration(
                    kind=validate_verification_stage_kind("test.stage"),
                    stage=stage,
                    required=True,
                ),
            ),
        ),
    )


def _identity_seed() -> DecisionFlowIdentitySeed:
    return DecisionFlowIdentitySeed(
        scope=DecisionScope(namespace="test", subject="case-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        decision_id=mint_decision_id(),
    )


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


def test_forbidden_patterns_absent() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source


def test_dual_authority_configuration_rejected() -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    with pytest.raises(DecisionCriticAuthorityConflictError):
        validate_decision_critic_authority_config(
            decision_gate=gate,
            verify_graph_final=True,
        )


@pytest.mark.asyncio
async def test_passed_flow_returns_continue(lifecycle_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=1),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.CONTINUE
    assert result.accepted_decision is not None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION


@pytest.mark.asyncio
async def test_challenged_with_revision_allowed_blocks_without_terminal_resolution(
    lifecycle_binding,
) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(ChallengedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=1),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="bad"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.revision_decision is not None
    assert result.revision_decision.disposition is DecisionRevisionDisposition.ALLOWED
    assert result.resolution_record is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.REVISION
    assert result.authority_reason == "decision_revision_required"


@pytest.mark.asyncio
async def test_revision_exhausted_requests_human_review(lifecycle_binding) -> None:
    port = RecordingHumanReviewPort()
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(ChallengedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            human_review_port=port,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="bad"),
            flow_scope=DecisionFlowScope.UAEP_STEP,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.PENDING_HUMAN
    assert result.human_review_pending is not None
    assert result.resolution_record is None
    assert port.pending is not None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.ADJUDICATION


@pytest.mark.asyncio
async def test_governance_deny_blocks_action_without_rejecting_accepted_decision(
    lifecycle_binding,
) -> None:
    payload = Payload(text="ok")
    identity_seed = _identity_seed()
    evaluator_spec = _governance_spec(
        DenyGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            governance_spec=evaluator_spec,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=identity_seed,
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=payload,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.authority_reason == "decision_governance_denied"
    assert result.accepted_decision is not None
    assert result.accepted_decision.identity == decision_identity_from_seed(identity_seed)
    assert result.accepted_decision.artifact == DecisionArtifact(
        kind=validate_decision_artifact_kind("test.payload"),
        content=payload,
    )
    assert result.authorization is None
    assert result.resolution_record is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION


@pytest.mark.asyncio
async def test_governance_allow_mints_authorization(lifecycle_binding) -> None:
    evaluator_spec = _governance_spec(
        AllowGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            governance_spec=evaluator_spec,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.CONTINUE
    assert result.accepted_decision is not None
    assert result.authorization is not None


def evaluator_spec_action():
    return decision_execution_action(
        kind=validate_decision_execution_action_kind("tool.notify"),
        subject="ops",
    )


def evaluator_spec_policy():
    return decision_governance_policy_context(
        policy_provenance_digest="digest-a",
        matched_rule_ids=("rule.allow",),
    )


@pytest.mark.asyncio
async def test_governance_require_human_with_port_pending(lifecycle_binding) -> None:
    port = RecordingHumanReviewPort()
    evaluator_spec = _governance_spec(
        RequireHumanGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    identity_seed = _identity_seed()
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            governance_spec=evaluator_spec,
            human_review_port=port,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=identity_seed,
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.UAEP_STEP,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.PENDING_HUMAN
    assert result.accepted_decision is not None
    assert result.authorization is None
    assert result.resolution_record is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION
    assert port.pending is not None
    assert port.pending.request.proposal_ref.identity == decision_identity_from_seed(
        identity_seed,
    )
    assert port.pending.request.reason_code == governance_requires_human_review_reason()


@pytest.mark.asyncio
async def test_governance_require_human_without_port_fails_closed(lifecycle_binding) -> None:
    evaluator_spec = _governance_spec(
        RequireHumanGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            governance_spec=evaluator_spec,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.UAEP_STEP,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.accepted_decision is not None
    assert result.authorization is None
    assert result.authority_reason == "decision_governance_human_review_unavailable"


@pytest.mark.asyncio
async def test_governance_require_human_transport_failure_fails_closed(
    lifecycle_binding,
) -> None:
    evaluator_spec = _governance_spec(
        RequireHumanGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            governance_spec=evaluator_spec,
            human_review_port=FailingHumanReviewPort(),
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.UAEP_STEP,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.accepted_decision is not None
    assert result.authorization is None
    assert result.authority_reason == "decision_governance_human_review_unavailable"


@pytest.mark.asyncio
async def test_revision_exhausted_without_human_terminal_rejected(lifecycle_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(ChallengedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            request_human_on_revision_exhausted=False,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="bad"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.resolution_record is not None
    assert result.resolution_record.resolution is DecisionResolution.REJECTED
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION
