# © Artur Czarnecki. All rights reserved.

"""Shared harness for Decision/Critic real capability parity qualification (DS-MIG-PARITY)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    DecisionGovernanceEvaluationInput,
    DecisionGovernancePolicyContext,
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    DecisionHumanReviewRequest,
)
from intergrax.contracts.decision_record import CandidateDecision
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_verification import (
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.domain_verification import (
    DomainVerificationOutcome,
    DomainVerifierId,
    domain_verification_failed,
    domain_verification_passed,
)
from intergrax.contracts.evidence_claims import (
    EvidenceBackedClaim,
    EvidenceClaimSet,
    mint_evidence_claim_id,
    validate_claim_kind,
    validate_evidence_reference_id,
)
from intergrax.contracts.evidence_verification import EvidenceClaimsProvider, EvidenceReferenceResolver
from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.contracts.guardrail_verification import assess_guardrail_scan
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticRubricRef,
    VerifierIndependenceMode,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
)
from intergrax.contracts.trajectory_verification import TrajectoryAgentId
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult
from intergrax.runtime.critic.contracts import RubricSpec
from intergrax.runtime.critic.critic_wiring import (
    CriticHookConfig,
    build_critic_graph_hooks,
    validate_final_with_critic_detail,
    validate_uaep_step_with_critic_detail,
)
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowGovernanceSpec,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import (
    agent_execution_decision_context,
    agent_execution_identity_seed,
    build_agent_execution_flow_request,
    evaluate_agent_execution_flow,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.domain import (
    DOMAIN_VERIFICATION_STAGE_KIND,
    IndependentDomainVerificationStage,
)
from intergrax.runtime.decision_verification_stages.evidence import (
    EVIDENCE_VERIFICATION_STAGE_KIND,
    EvidenceVerificationStage,
)
from intergrax.runtime.decision_verification_stages.guardrail import (
    GUARDRAIL_VERIFICATION_STAGE_KIND,
    GuardrailVerificationStage,
)
from intergrax.runtime.decision_verification_stages.semantic import (
    SEMANTIC_VERIFICATION_STAGE_KIND,
    SemanticVerificationStage,
)
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    AgentExecutionStructuralValidator,
    StructuralVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
    TrajectoryVerificationStage,
)
from intergrax.runtime.migration.critic_shadow_adapter import (
    CriticShadowAdapter,
    CriticShadowConfig,
    build_critic_shadow_adapter,
    evaluate_critic_shadow_and_compare,
)
from intergrax.runtime.migration.decision_critic_parity import (
    DecisionCriticParityClassification,
    DecisionCriticParityResult,
    ParityHostScope,
    ParityVerificationCapability,
    compare_decision_critic_parity,
)
from intergrax.tools.providers.eval.contracts import (
    EvalJudgeInput,
    EvalJudgeOutput,
    EvalTrajectoryInput,
    EvalTrajectoryOutput,
)

GUARDRAIL_SCAN_KEY = "guardrail_scan"
PARITY_EVIDENCE_REF_KEY = "parity_evidence_ref"
DOMAIN_INVALID_MARKER = "domain-invalid"
SEMANTIC_FAIL_MARKER = "semantic-fail"
TRAJECTORY_FAIL_MARKER = "trajectory-fail"

_QUALIFICATION_RUBRIC = resolved_semantic_rubric(
    ref=semantic_rubric_ref(rubric_id="parity.qualification", version=1),
    criteria=("Output must be acceptable.",),
    min_score=0.75,
    provenance_ref="prompt_registry:parity.qualification@1",
)
_DOMAIN_REQUIREMENT = validate_verification_requirement_code(
    "verification.domain.requirement_failed",
)
_DOMAIN_FINDING = validate_verification_finding_code(
    "verification.domain.requirement_failed",
)
_KNOWN_EVIDENCE_REF = validate_evidence_reference_id("parity.evidence.ref.1")


class ParityQualificationMode(str, Enum):
    """How one qualification case exercises retirement evidence."""

    CROSS_SYSTEM = "cross_system"
    DECISION_SUPERSET = "decision_superset"
    ARCHITECTURAL_MAPPING = "architectural_mapping"


@dataclass(frozen=True, slots=True)
class ParityQualificationCase:
    """Typed qualification case contract."""

    case_id: str
    scope: ParityHostScope
    capability: ParityVerificationCapability
    expected_classification: DecisionCriticParityClassification
    mode: ParityQualificationMode


@dataclass(frozen=True, slots=True)
class AgentExecutionSemanticContentProvider:
    def extract(self, candidate: CandidateDecision[AgentExecutionResult]) -> str:
        return candidate.artifact.content.summary


@dataclass(frozen=True, slots=True)
class AgentExecutionTrajectoryAgentProvider:
    def resolve(self, candidate: CandidateDecision[AgentExecutionResult]) -> TrajectoryAgentId:
        return TrajectoryAgentId(candidate.artifact.content.agent_id)


@dataclass(frozen=True, slots=True)
class AgentExecutionGuardrailScanProvider:
    def extract(self, candidate: CandidateDecision[AgentExecutionResult]) -> GuardrailScanResult | None:
        structured = candidate.artifact.content.structured_data
        raw = structured.get(GUARDRAIL_SCAN_KEY)
        if not isinstance(raw, dict):
            return None
        allowed = raw.get("allowed")
        resolved_allowed = True if allowed is None else bool(allowed)
        detail = str(raw.get("detail") or "")
        categories_raw = raw.get("categories")
        categories: tuple[str, ...] = ()
        if isinstance(categories_raw, (list, tuple)):
            categories = tuple(str(category) for category in categories_raw)
        return GuardrailScanResult(
            allowed=resolved_allowed,
            detail=detail,
            categories=categories,
        )


@dataclass(frozen=True, slots=True)
class AgentExecutionEvidenceClaimsProvider:
    def extract(self, candidate: CandidateDecision[AgentExecutionResult]) -> EvidenceClaimSet | None:
        structured = candidate.artifact.content.structured_data
        ref_value = structured.get(PARITY_EVIDENCE_REF_KEY)
        if ref_value is None:
            return None
        ref = validate_evidence_reference_id(str(ref_value))
        claim = EvidenceBackedClaim(
            claim_id=mint_evidence_claim_id(),
            statement="Parity qualification claim.",
            claim_kind=validate_claim_kind("generic.claim"),
            supporting_evidence_ids=(ref,),
        )
        return EvidenceClaimSet(claims=(claim,))


@dataclass(frozen=True, slots=True)
class InMemoryEvidenceReferenceResolver:
    known_ids: frozenset[str]
    available: bool = True

    def is_available(self) -> bool:
        return self.available

    def evidence_exists(self, evidence_id: object) -> bool:
        return str(evidence_id) in self.known_ids


@dataclass(frozen=True, slots=True)
class FixedSemanticRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        return self.rubric


@dataclass(frozen=True, slots=True)
class DeterministicSemanticJudge:
    """Fail when content contains ``semantic-fail``."""

    available: bool = True

    def is_available(self) -> bool:
        return self.available

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        passed = SEMANTIC_FAIL_MARKER not in params.output_text
        return EvalJudgeOutput(
            rubric_id=params.rubric_id,
            score=1.0 if passed else 0.2,
            passed=passed,
            reasons=[] if passed else ["below threshold"],
        )


@dataclass(frozen=True, slots=True)
class DeterministicTrajectoryEvaluator:
    """Fail when tenant_id contains ``trajectory-fail``."""

    available: bool = True

    def is_available(self) -> bool:
        return self.available

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        passed = TRAJECTORY_FAIL_MARKER not in params.tenant_id
        return EvalTrajectoryOutput(
            run_id=params.run_id,
            score=1.0 if passed else 0.2,
            passed=passed,
            reasons=[] if passed else ["below threshold"],
        )


@dataclass(frozen=True, slots=True)
class SharedEvalToolClient:
    """One deterministic eval source for Decision semantic and Critic L1."""

    judge_impl: DeterministicSemanticJudge
    trajectory_impl: DeterministicTrajectoryEvaluator

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        return self.judge_impl.judge(params)

    def trajectory(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        return self.trajectory_impl.evaluate(params)


@dataclass(frozen=True, slots=True)
class SummaryDomainVerifier:
    verifier_id_value: DomainVerifierId

    @property
    def verifier_id(self) -> DomainVerifierId:
        return self.verifier_id_value

    def is_available(self) -> bool:
        return True

    def verify(self, candidate: CandidateDecision[AgentExecutionResult]) -> DomainVerificationOutcome:
        if DOMAIN_INVALID_MARKER in candidate.artifact.content.summary:
            return domain_verification_failed(
                requirement_code=_DOMAIN_REQUIREMENT,
                finding_code=_DOMAIN_FINDING,
                message="domain requirement failed",
            )
        return domain_verification_passed()


class QualificationRequireHumanGovernanceEvaluator:
    def __init__(
        self,
        *,
        action: DecisionExecutionAction,
        policy_context: DecisionGovernancePolicyContext,
    ) -> None:
        self._action = action
        self._policy_context = policy_context

    def evaluate(self, *, evaluation_input: DecisionGovernanceEvaluationInput) -> DecisionGovernanceDecision:
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.REQUIRE_HUMAN,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self._action,
            policy_context=self._policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


def _qualification_governance_spec() -> DecisionFlowGovernanceSpec[AgentExecutionResult]:
    action = decision_execution_action(
        kind=validate_decision_execution_action_kind("tool.notify"),
        subject="ops",
    )
    policy_context = decision_governance_policy_context(
        policy_provenance_digest="parity-qualification",
        matched_rule_ids=("rule.require_human",),
    )
    return DecisionFlowGovernanceSpec(
        action=action,
        policy_context=policy_context,
        evaluator=QualificationRequireHumanGovernanceEvaluator(
            action=action,
            policy_context=policy_context,
        ),
    )


class RecordingDecisionHumanReviewPort:
    """No-op human review port that records pending review requests."""

    def __init__(self) -> None:
        self.pending: DecisionHumanReviewPending | None = None
        self.requests: list[DecisionHumanReviewRequest] = []

    def request_review(self, request: DecisionHumanReviewRequest) -> DecisionHumanReviewPending:
        from intergrax.runtime.decision_human_review import request_decision_human_review

        self.requests.append(request)
        self.pending = request_decision_human_review(request)
        return self.pending


@dataclass(frozen=True, slots=True)
class QualificationPipelineOptions:
    include_guardrail: bool = False
    include_semantic: bool = False
    include_trajectory: bool = False
    include_evidence: bool = False
    include_domain: bool = False
    semantic_required: bool = True
    semantic_judge: DeterministicSemanticJudge | None = None
    trajectory_evaluator: DeterministicTrajectoryEvaluator | None = None
    evidence_valid: bool = True


def _parity_contract() -> AgentContract:
    return AgentContract(
        id="parity-agent",
        name="parity-agent",
        description="parity qualification",
        validation_rules=["non_empty_summary"],
    )


def _build_pipeline(
    *,
    contract: AgentContract,
    options: QualificationPipelineOptions,
) -> VerificationPipeline[AgentExecutionResult]:
    registrations: list[VerificationStageRegistration[AgentExecutionResult]] = [
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=StructuralVerificationStage(
                validators=(
                    AgentExecutionStructuralValidator(contract=contract),
                ),
            ),
            required=True,
        ),
    ]
    if options.include_guardrail:
        registrations.append(
            VerificationStageRegistration(
                kind=GUARDRAIL_VERIFICATION_STAGE_KIND,
                stage=GuardrailVerificationStage(
                    scan_provider=AgentExecutionGuardrailScanProvider(),
                ),
                required=True,
            ),
        )
    judge = options.semantic_judge if options.semantic_judge is not None else DeterministicSemanticJudge()
    if options.include_semantic:
        registrations.append(
            VerificationStageRegistration(
                kind=SEMANTIC_VERIFICATION_STAGE_KIND,
                stage=SemanticVerificationStage(
                    rubric_ref=_QUALIFICATION_RUBRIC.ref,
                    rubric_resolver=FixedSemanticRubricResolver(rubric=_QUALIFICATION_RUBRIC),
                    content_provider=AgentExecutionSemanticContentProvider(),
                    judge=judge,
                    independence=semantic_verification_independence_config(
                        mode=VerifierIndependenceMode.INDEPENDENT,
                        producer_profile_id="producer",
                        verifier_profile_id="verifier",
                    ),
                ),
                required=options.semantic_required,
            ),
        )
    trajectory = (
        options.trajectory_evaluator
        if options.trajectory_evaluator is not None
        else DeterministicTrajectoryEvaluator()
    )
    if options.include_trajectory:
        registrations.append(
            VerificationStageRegistration(
                kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
                stage=TrajectoryVerificationStage(
                    evaluator=trajectory,
                    agent_id_provider=AgentExecutionTrajectoryAgentProvider(),
                ),
                required=True,
            ),
        )
    if options.include_evidence:
        known = frozenset({str(_KNOWN_EVIDENCE_REF)}) if options.evidence_valid else frozenset()
        registrations.append(
            VerificationStageRegistration(
                kind=EVIDENCE_VERIFICATION_STAGE_KIND,
                stage=EvidenceVerificationStage(
                    claims_provider=AgentExecutionEvidenceClaimsProvider(),
                    resolver=InMemoryEvidenceReferenceResolver(known_ids=known),
                ),
                required=True,
            ),
        )
    if options.include_domain:
        registrations.append(
            VerificationStageRegistration(
                kind=DOMAIN_VERIFICATION_STAGE_KIND,
                stage=IndependentDomainVerificationStage(
                    verifier=SummaryDomainVerifier(
                        verifier_id_value=DomainVerifierId("domain.parity"),
                    ),
                    execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                ),
                required=True,
            ),
        )
    return VerificationPipeline(registry=verification_stage_registry(tuple(registrations)))


def _build_gate(
    *,
    contract: AgentContract,
    flow_scope: DecisionFlowScope,
    options: QualificationPipelineOptions,
    human_review_port: RecordingDecisionHumanReviewPort | None = None,
    max_revisions: int = 0,
    governance_spec: DecisionFlowGovernanceSpec[AgentExecutionResult] | None = None,
) -> CanonicalDecisionFlowGate[AgentExecutionResult]:
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_build_pipeline(contract=contract, options=options),
            revision_policy=decision_revision_policy(max_revisions=max_revisions),
            scopes=frozenset({flow_scope}),
            human_review_port=human_review_port,
            governance_spec=governance_spec,
        ),
    )


def _execution(
    *,
    run_id: str,
    summary: str,
    guardrail_scan: GuardrailScanResult | None = None,
    evidence_ref: str | None = None,
) -> AgentExecutionResult:
    structured: dict[str, object] = {}
    if guardrail_scan is not None:
        structured[GUARDRAIL_SCAN_KEY] = {
            "allowed": guardrail_scan.allowed,
            "detail": guardrail_scan.detail,
            "categories": list(guardrail_scan.categories),
        }
    if evidence_ref is not None:
        structured[PARITY_EVIDENCE_REF_KEY] = evidence_ref
    return AgentExecutionResult(
        agent_id="parity-agent",
        run_id=run_id,
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
        structured_data=structured,
    )


def _guardrail_critic_context(scan: GuardrailScanResult | None) -> dict[str, object]:
    if scan is None:
        return {}
    return {
        GUARDRAIL_SCAN_KEY: {
            "allowed": scan.allowed,
            "detail": scan.detail,
            "categories": list(scan.categories),
        },
    }


def _shadow_config(
    *,
    semantic: bool = False,
    trajectory: bool = False,
) -> CriticShadowConfig:
    return CriticShadowConfig(
        semantic_judge_enabled=semantic,
        trajectory_eval_enabled=trajectory,
        judge_threshold=0.75,
        default_rubric_ref=str(_QUALIFICATION_RUBRIC.ref.rubric_id),
    )


def build_qualification_shadow(
    *,
    eval_client: CriticEvalToolClient | None,
    semantic: bool = False,
    trajectory: bool = False,
) -> CriticShadowAdapter:
    return build_critic_shadow_adapter(
        config=_shadow_config(semantic=semantic, trajectory=trajectory),
        l1_client=eval_client,
    )


async def run_graph_parity_case(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    tenant_id: str,
    subject: str,
    summary: str,
    pipeline_options: QualificationPipelineOptions,
    shadow: CriticShadowAdapter,
    guardrail_scan: GuardrailScanResult | None = None,
    evidence_ref: str | None = None,
    trajectory_tenant_id: str | None = None,
) -> DecisionCriticParityResult:
    contract = _parity_contract()
    resolved_tenant_id = trajectory_tenant_id if trajectory_tenant_id is not None else tenant_id
    gate = _build_gate(
        contract=contract,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        options=pipeline_options,
    )
    execution = _execution(
        run_id=str(run_id),
        summary=summary,
        guardrail_scan=guardrail_scan,
        evidence_ref=evidence_ref,
    )
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=resolved_tenant_id,
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="graph.final",
        subject=subject,
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    critic_context = _guardrail_critic_context(guardrail_scan)
    return await evaluate_critic_shadow_and_compare(
        shadow=shadow,
        decision_result=flow_result,
        execution=execution,
        contract=contract,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id=resolved_tenant_id,
        subject=subject,
        extra_context=critic_context or None,
    )


async def run_uaep_parity_case(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    tenant_id: str,
    step_id: str,
    summary: str,
    pipeline_options: QualificationPipelineOptions,
    shadow: CriticShadowAdapter,
    guardrail_scan: GuardrailScanResult | None = None,
) -> DecisionCriticParityResult:
    contract = _parity_contract()
    gate = _build_gate(
        contract=contract,
        flow_scope=DecisionFlowScope.UAEP_STEP,
        options=pipeline_options,
    )
    execution = _execution(
        run_id=str(run_id),
        summary=summary,
        guardrail_scan=guardrail_scan,
    )
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="uaep.step",
        subject=step_id,
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.UAEP_STEP,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    return await evaluate_critic_shadow_and_compare(
        shadow=shadow,
        decision_result=flow_result,
        execution=execution,
        contract=contract,
        flow_scope=DecisionFlowScope.UAEP_STEP,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id=tenant_id,
        subject=step_id,
        step_id=step_id,
        extra_context=_guardrail_critic_context(guardrail_scan) or None,
    )


async def run_hitl_architectural_mapping_case(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    tenant_id: str,
    subject: str,
    flow_scope: DecisionFlowScope,
) -> DecisionCriticParityResult:
    contract = _parity_contract()
    human_port = RecordingDecisionHumanReviewPort()
    gate = _build_gate(
        contract=contract,
        flow_scope=flow_scope,
        options=QualificationPipelineOptions(),
        human_review_port=human_port,
        governance_spec=_qualification_governance_spec(),
    )
    execution = _execution(run_id=str(run_id), summary="valid summary")
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )
    namespace = "graph.final" if flow_scope is DecisionFlowScope.GRAPH_FINAL else "uaep.step"
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace=namespace,
        subject=subject,
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=flow_scope,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(
            verify_graph_final=flow_scope is DecisionFlowScope.GRAPH_FINAL,
            verify_uaep_step=flow_scope is DecisionFlowScope.UAEP_STEP,
            l2_human_required=True,
        ),
    )
    if hooks is None:
        raise RuntimeError("critic hooks required for HITL qualification")
    if flow_scope is DecisionFlowScope.GRAPH_FINAL:
        _, critic_verdict = validate_final_with_critic_detail(
            execution,
            contract=contract,
            hooks=hooks,
            task_id=str(task_id),
            run_id=str(run_id),
            tenant_id=tenant_id,
        )
    else:
        _, critic_verdict = validate_uaep_step_with_critic_detail(
            execution,
            contract=contract,
            hooks=hooks,
            task_id=str(task_id),
            run_id=str(run_id),
            tenant_id=tenant_id,
            step_id=subject,
        )
    from intergrax.runtime.migration.decision_critic_parity import build_parity_identity

    identity = build_parity_identity(
        flow_scope=flow_scope,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id=tenant_id,
        agent_id=contract.id,
        subject=subject,
        decision_result=flow_result,
    )
    return compare_decision_critic_parity(
        identity=identity,
        decision_result=flow_result,
        critic_verdict=critic_verdict,
    )


async def run_semantic_shadow_unavailable_case(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    tenant_id: str,
    subject: str,
) -> DecisionCriticParityResult:
    contract = _parity_contract()
    gate = _build_gate(
        contract=contract,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        options=QualificationPipelineOptions(include_semantic=True),
    )
    shadow = build_qualification_shadow(eval_client=None, semantic=True)
    execution = _execution(run_id=str(run_id), summary="valid summary")
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="graph.final",
        subject=subject,
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    return await evaluate_critic_shadow_and_compare(
        shadow=shadow,
        decision_result=flow_result,
        execution=execution,
        contract=contract,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id=tenant_id,
        subject=subject,
    )


def guardrail_scan_allowed() -> GuardrailScanResult:
    return GuardrailScanResult(allowed=True, detail="ok", categories=("safe",))


def guardrail_scan_blocked() -> GuardrailScanResult:
    return GuardrailScanResult(allowed=False, detail="output blocked")


def assess_shared_guardrail(scan: GuardrailScanResult) -> bool:
    return assess_guardrail_scan(scan).passed


def qualification_eval_client() -> SharedEvalToolClient:
    return SharedEvalToolClient(
        judge_impl=DeterministicSemanticJudge(),
        trajectory_impl=DeterministicTrajectoryEvaluator(),
    )


KNOWN_EVIDENCE_REF = _KNOWN_EVIDENCE_REF


def qualification_rubric_spec() -> RubricSpec:
    return RubricSpec(
        rubric_id=str(_QUALIFICATION_RUBRIC.ref.rubric_id),
        criteria=list(_QUALIFICATION_RUBRIC.criteria),
        reference_context=_QUALIFICATION_RUBRIC.reference_context,
        min_score=_QUALIFICATION_RUBRIC.min_score,
    )
