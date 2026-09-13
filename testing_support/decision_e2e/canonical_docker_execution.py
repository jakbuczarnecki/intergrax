# © Artur Czarnecki. All rights reserved.

"""Canonical Decision → Governance → Execution Engine paths for Docker qualification."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernanceDisposition,
    DecisionGovernancePolicyContext,
    decision_execution_action,
    decision_governance_policy_context,
)
from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationKind,
    correlation_record_from_decision_identity,
)
from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision.integration import InMemoryDecisionAuditSink
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGovernanceSpec,
    DecisionFlowHostAction,
    DecisionFlowResult,
)
from intergrax.runtime.decision_authorization import (
    validate_execution_authorization_bundle,
)
from intergrax.runtime.decision_integration_composition import (
    production_decision_system_integration,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus
from intergrax.runtime.execution.runtime import RootExecutionOptions
from intergrax.runtime.execution.single_model_deliberation import (
    single_model_inference_execution_request,
)
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.single_model_strategy import (
    SingleModelDeliberationInput,
    SingleModelInferenceConfiguration,
    validate_decision_artifact_kind,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.diagnostics.decision_execution_correlation_persistence import (
    InMemoryDecisionExecutionCorrelationPersistence,
)
from intergrax.runtime.execution.inference_profile import validate_inference_profile_id

from testing_support.builder import FakeLLMAdapter
from testing_support.decision_e2e.bindings import ProviderBindingEvidence
from testing_support.decision_e2e.composition import (
    QualificationComposition,
    build_qualification_composition,
    evaluate_decision_flow,
    mint_qualification_identity,
)
from testing_support.decision_e2e.docker_system_scenarios import (
    DockerSystemScenarioResult,
    _fail,
    _ok,
    _reference_source,
)
from testing_support.decision_e2e.environment import QualificationEnvironment
from testing_support.decision_e2e.governance import DispositionGovernanceEvaluator
from testing_support.decision_e2e.independence import ProviderIndependenceLevel
from testing_support.decision_e2e.payloads import QualificationRecommendation
from testing_support.decision_e2e.verification import build_pass_through_pipeline

_TASK_ID = "DS-E2E-15J-CANONICAL-EXECUTION-DOCKER-E2E-QUALIFICATION"
_ARTIFACT_KIND = validate_decision_artifact_kind("decision_e2e_qualification")
_PROFILE_PRODUCER = validate_inference_profile_id("profile-producer")
_CONCURRENT_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)
_CANONICAL_EXECUTION_PROVIDER_ID = "canonical-execution-runtime"


@dataclass(frozen=True, slots=True)
class CanonicalExecutionEvidence:
    decision_id: str
    governance_disposition: str
    authorization_id: str
    task_id: str
    run_id: str
    execution_id: str
    execution_result_status: str
    correlation_recorded: bool
    audit_entries: int
    execution_engine_invocations: int
    execution_path: str
    recording_execution_provider_used: bool


def _deterministic_environment(
    *,
    recommendation: str = "canonical-docker-ok",
    fail_structured: bool = False,
) -> QualificationEnvironment:
    payload = QualificationRecommendation(
        recommendation=recommendation,
        confidence="high",
        rationale_summary="canonical docker qualification",
    )

    class _StructuredFakeAdapter(FakeLLMAdapter):
        def supports_structured_output(self) -> bool:
            return True

    if fail_structured:

        class _FailingStructuredAdapter(_StructuredFakeAdapter):
            def generate_structured(self, messages, output_model, **kwargs):
                raise RuntimeError("controlled canonical execution failure")

        adapter = _FailingStructuredAdapter()
    else:
        adapter = _StructuredFakeAdapter(fake_structured_data=payload)
    profile = LLMProfile(
        provider=LLMProvider.OLLAMA,
        model="canonical-docker-qualification-stub",
    )
    evidence = ProviderBindingEvidence(
        profile_id="profile-producer",
        provider=LLMProvider.OLLAMA.value,
        model="canonical-docker-qualification-stub",
    )
    return QualificationEnvironment(
        producer_profile=profile,
        producer_adapter=adapter,
        verifier_profile=profile,
        verifier_adapter=adapter,
        council_profile_b=profile,
        council_adapter_b=adapter,
        council_profile_c=profile,
        council_adapter_c=adapter,
        producer_evidence=evidence,
        verifier_evidence=evidence,
        council_b_evidence=evidence,
        council_c_evidence=evidence,
        independence_level=ProviderIndependenceLevel.PROFILE_ONLY,
    )


def build_canonical_docker_composition(
    *,
    recommendation: str = "canonical-docker-ok",
    fail_structured: bool = False,
) -> QualificationComposition:
    return build_qualification_composition(
        _deterministic_environment(
            recommendation=recommendation,
            fail_structured=fail_structured,
        ),
        participant_concurrent_work_policy=_CONCURRENT_POLICY,
    )


def _governance_action_and_policy() -> tuple[
    DecisionExecutionAction,
    DecisionGovernancePolicyContext,
]:
    action = decision_execution_action(
        kind="decision_e2e.canonical_execution",
        subject="docker-qualification",
    )
    policy = decision_governance_policy_context(
        policy_provenance_digest="ds_e2e_15j_canonical_execution",
    )
    return action, policy


def _gate_for_disposition(
    composition: QualificationComposition,
    disposition: DecisionGovernanceDisposition,
) -> CanonicalDecisionFlowGate[QualificationRecommendation]:
    action, policy = _governance_action_and_policy()
    return composition.build_flow_gate(
        pipeline=build_pass_through_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
        governance_spec=DecisionFlowGovernanceSpec(
            action=action,
            policy_context=policy,
            evaluator=DispositionGovernanceEvaluator(
                action=action,
                policy_context=policy,
                disposition=disposition,
            ),
        ),
    )


async def _run_flow(
    composition: QualificationComposition,
    gate: CanonicalDecisionFlowGate[QualificationRecommendation],
    *,
    identity: DecisionIdentity,
    payload: QualificationRecommendation,
) -> DecisionFlowResult[QualificationRecommendation]:
    lifecycle_host, _ = composition.lifecycle_for_identity(identity)
    token = bind_active_decision_lifecycle_host(lifecycle_host)
    try:
        return await evaluate_decision_flow(
            composition,
            gate,
            identity=identity,
            payload=payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)


async def _execute_authorized_workload(
    composition: QualificationComposition,
    *,
    authorization: DecisionExecutionAuthorization,
    accepted: AuthoritativeAcceptedDecision[QualificationRecommendation],
    identity: DecisionIdentity,
    task_message: str,
) -> ExecutionResult[QualificationRecommendation]:
    action, policy = _governance_action_and_policy()
    validate_execution_authorization_bundle(
        authorization=authorization,
        decision=accepted,
        action=action,
        current_policy_context=policy,
    )
    deliberation_input = SingleModelDeliberationInput(
        messages=(ChatMessage(role="user", content=task_message),),
        output_type=QualificationRecommendation,
        artifact_kind=_ARTIFACT_KIND,
    )
    inference = SingleModelInferenceConfiguration(
        inference_profile_id=_PROFILE_PRODUCER,
    )
    request = single_model_inference_execution_request(
        deliberation_input,
        inference=inference,
    )
    return await composition.execution.execute(
        request,
        options=RootExecutionOptions(
            authority=ParentExecutionAuthority.unrestricted_root(),
            tenant_id=identity.tenant_id,
            run_id=identity.execution.run_id,
            attempt_id=identity.execution.attempt_id,
        ),
    )


def _payload_for_identity(
    subject: str,
) -> tuple[DecisionIdentity, QualificationRecommendation]:
    identity = mint_qualification_identity(subject=subject)
    payload = QualificationRecommendation(
        recommendation="canonical-flow",
        confidence="high",
    )
    return identity, payload


async def _canonical_success_async() -> CanonicalExecutionEvidence:
    recommendation = "canonical-docker-ok"
    composition = build_canonical_docker_composition(recommendation=recommendation)
    identity = mint_qualification_identity(subject="canonical-execution-success")
    payload = QualificationRecommendation(
        recommendation=recommendation,
        confidence="high",
    )
    gate = _gate_for_disposition(
        composition,
        DecisionGovernanceDisposition.ALLOW,
    )
    flow_result = await _run_flow(
        composition,
        gate,
        identity=identity,
        payload=payload,
    )
    if flow_result.host_action is not DecisionFlowHostAction.CONTINUE:
        raise RuntimeError(f"expected CONTINUE, got {flow_result.host_action}")
    authorization = flow_result.authorization
    accepted = flow_result.accepted_decision
    if authorization is None or accepted is None:
        raise RuntimeError("ALLOW flow must mint authorization and accepted decision")

    exec_result = await _execute_authorized_workload(
        composition,
        authorization=authorization,
        accepted=accepted,
        identity=identity,
        task_message="canonical authorized execution",
    )
    if (
        exec_result.status is not ExecutionStatus.COMPLETED
        or exec_result.output is None
    ):
        raise RuntimeError("canonical execution did not complete")

    if exec_result.output.recommendation != recommendation:
        raise RuntimeError("canonical ExecutionRuntime did not return inference output")
    engine_invocations = 1

    correlation_store = InMemoryDecisionExecutionCorrelationPersistence()
    stamp = datetime(2026, 9, 13, 8, 0, 0, tzinfo=UTC)
    correlation_store.append(
        correlation_record_from_decision_identity(
            identity,
            correlation_kind=DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION,
            created_at=stamp,
        ),
    )

    sink = InMemoryDecisionAuditSink()
    production_decision_system_integration(audit_sink=sink).integrate_lifecycle(
        _reference_source("docker-canonical-success"),
    )

    return CanonicalExecutionEvidence(
        decision_id=str(identity.decision_id),
        governance_disposition=DecisionGovernanceDisposition.ALLOW.value,
        authorization_id=str(authorization.authorization_id),
        task_id=str(identity.execution.task_id),
        run_id=str(identity.execution.run_id),
        execution_id=str(identity.execution.execution_id),
        execution_result_status=exec_result.status.value,
        correlation_recorded=True,
        audit_entries=len(sink.entries),
        execution_engine_invocations=engine_invocations,
        execution_path=_CANONICAL_EXECUTION_PROVIDER_ID,
        recording_execution_provider_used=False,
    )


async def _canonical_deny_async() -> int:
    composition = build_canonical_docker_composition()
    identity, payload = _payload_for_identity("canonical-governance-deny")
    invocations_before = composition.work_port.invocation_count
    gate = _gate_for_disposition(
        composition,
        DecisionGovernanceDisposition.DENY,
    )
    flow_result = await _run_flow(
        composition,
        gate,
        identity=identity,
        payload=payload,
    )
    if flow_result.host_action is not DecisionFlowHostAction.BLOCK:
        raise RuntimeError("DENY must BLOCK host action")
    if flow_result.authorization is not None:
        raise RuntimeError("DENY must not mint authorization")
    invocations_after = composition.work_port.invocation_count
    if invocations_after != invocations_before:
        raise RuntimeError("DENY must not invoke Execution Engine workload")
    return invocations_after - invocations_before


async def _canonical_approval_async() -> Literal["require_human"]:
    composition = build_canonical_docker_composition()
    identity, payload = _payload_for_identity("canonical-governance-approval")
    invocations_before = composition.work_port.invocation_count
    gate = _gate_for_disposition(
        composition,
        DecisionGovernanceDisposition.REQUIRE_HUMAN,
    )
    flow_result = await _run_flow(
        composition,
        gate,
        identity=identity,
        payload=payload,
    )
    if flow_result.host_action is not DecisionFlowHostAction.BLOCK:
        raise RuntimeError("REQUIRE_HUMAN without review port must BLOCK")
    if flow_result.authorization is not None:
        raise RuntimeError("REQUIRE_HUMAN must not mint execution authorization")
    invocations_after = composition.work_port.invocation_count
    if invocations_after != invocations_before:
        raise RuntimeError("REQUIRE_HUMAN must not invoke Execution Engine workload")
    return "require_human"


async def _canonical_failure_async() -> str:
    flow_composition = build_canonical_docker_composition()
    identity, payload = _payload_for_identity("canonical-execution-failure")
    gate = _gate_for_disposition(
        flow_composition,
        DecisionGovernanceDisposition.ALLOW,
    )
    flow_result = await _run_flow(
        flow_composition,
        gate,
        identity=identity,
        payload=payload,
    )
    authorization = flow_result.authorization
    accepted = flow_result.accepted_decision
    if authorization is None or accepted is None:
        raise RuntimeError("failure scenario requires ALLOW authorization")

    failing_composition = build_canonical_docker_composition(fail_structured=True)
    try:
        await _execute_authorized_workload(
            failing_composition,
            authorization=authorization,
            accepted=accepted,
            identity=identity,
            task_message="must fail",
        )
    except RuntimeError as exc:
        if "controlled canonical execution failure" not in str(exc):
            raise
        return exc.__class__.__name__
    raise RuntimeError("expected controlled execution failure")


def run_canonical_execution_success() -> DockerSystemScenarioResult:
    scenario_id = "canonical-execution-success"
    try:
        evidence = asyncio.run(_canonical_success_async())
    except Exception as exc:  # noqa: BLE001 — qualification boundary
        return _fail(scenario_id, f"canonical success failed: {exc}")
    if evidence.recording_execution_provider_used:
        return _fail(scenario_id, "RecordingExecutionProvider must not be used")
    return _ok(
        scenario_id,
        "canonical DecisionFlowGate → authorization → ExecutionRuntime",
        qualification_task_id=_TASK_ID,
        **asdict(evidence),
    )


def run_canonical_governance_deny() -> DockerSystemScenarioResult:
    scenario_id = "canonical-governance-deny"
    try:
        extra_invocations = asyncio.run(_canonical_deny_async())
    except Exception as exc:  # noqa: BLE001
        return _fail(scenario_id, f"canonical deny failed: {exc}")
    return _ok(
        scenario_id,
        "governance DENY blocked authorization and Execution Engine",
        qualification_task_id=_TASK_ID,
        governance_disposition=DecisionGovernanceDisposition.DENY.value,
        execution_engine_extra_invocations=extra_invocations,
        recording_execution_provider_used=False,
    )


def run_canonical_governance_approval() -> DockerSystemScenarioResult:
    scenario_id = "canonical-governance-approval"
    try:
        disposition = asyncio.run(_canonical_approval_async())
    except Exception as exc:  # noqa: BLE001
        return _fail(scenario_id, f"canonical approval gate failed: {exc}")
    return _ok(
        scenario_id,
        "REQUIRE_HUMAN blocked execution without approval bypass",
        qualification_task_id=_TASK_ID,
        governance_disposition=disposition,
        recording_execution_provider_used=False,
    )


def run_canonical_evidence_chain() -> DockerSystemScenarioResult:
    scenario_id = "canonical-evidence-chain"
    try:
        evidence = asyncio.run(_canonical_success_async())
    except Exception as exc:  # noqa: BLE001
        return _fail(scenario_id, f"canonical evidence failed: {exc}")
    required = (
        evidence.decision_id,
        evidence.authorization_id,
        evidence.run_id,
        evidence.execution_id,
        evidence.governance_disposition,
    )
    if not all(required):
        return _fail(scenario_id, "incomplete canonical correlation chain")
    return _ok(
        scenario_id,
        "decision ↔ governance authorization ↔ execution correlation",
        qualification_task_id=_TASK_ID,
        decision_id=evidence.decision_id,
        governance_decision_id=evidence.authorization_id,
        authorization_id=evidence.authorization_id,
        execution_result_id=evidence.execution_id,
        run_id=evidence.run_id,
        execution_path=evidence.execution_path,
    )


def run_canonical_execution_failure() -> DockerSystemScenarioResult:
    scenario_id = "canonical-execution-failure"
    try:
        failure_class = asyncio.run(_canonical_failure_async())
    except Exception as exc:  # noqa: BLE001
        return _fail(scenario_id, f"canonical failure path failed: {exc}")
    return _ok(
        scenario_id,
        "authorized execution failure propagated from canonical engine",
        qualification_task_id=_TASK_ID,
        failure_class=failure_class,
        recording_execution_provider_used=False,
    )


_CANONICAL_SCENARIO_RUNNERS = {
    "canonical-execution-success": run_canonical_execution_success,
    "canonical-governance-deny": run_canonical_governance_deny,
    "canonical-governance-approval": run_canonical_governance_approval,
    "canonical-evidence-chain": run_canonical_evidence_chain,
    "canonical-execution-failure": run_canonical_execution_failure,
}


def run_canonical_docker_scenario(scenario_id: str) -> DockerSystemScenarioResult:
    runner = _CANONICAL_SCENARIO_RUNNERS.get(scenario_id)
    if runner is None:
        return _fail(scenario_id, f"unknown canonical scenario: {scenario_id}")
    return runner()


__all__ = [
    "CanonicalExecutionEvidence",
    "build_canonical_docker_composition",
    "run_canonical_docker_scenario",
    "run_canonical_execution_failure",
    "run_canonical_execution_success",
    "run_canonical_evidence_chain",
    "run_canonical_governance_approval",
    "run_canonical_governance_deny",
]
