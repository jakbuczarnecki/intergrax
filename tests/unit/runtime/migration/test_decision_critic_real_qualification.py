# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.decision_flow import DecisionFlowScope
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    DECISION_SUPERSET_CAPABILITY,
    DecisionCriticParityClassification,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    LEGACY_HITL_IS_DECISION_HUMAN_REVIEW,
    ParityHostScope,
    ParityVerificationCapability,
    aggregate_parity_metrics,
    evaluate_critic_retirement_readiness,
)
from testing_support.decision_critic_parity_qualification import (
    DOMAIN_INVALID_MARKER,
    KNOWN_EVIDENCE_REF,
    SEMANTIC_FAIL_MARKER,
    TRAJECTORY_FAIL_MARKER,
    DeterministicSemanticJudge,
    ParityQualificationCase,
    ParityQualificationMode,
    QualificationPipelineOptions,
    build_qualification_shadow,
    guardrail_scan_allowed,
    guardrail_scan_blocked,
    qualification_eval_client,
    run_graph_parity_case,
    run_hitl_architectural_mapping_case,
    run_semantic_shadow_unavailable_case,
    run_uaep_parity_case,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SUPPORT_PATH = Path("testing_support/decision_critic_parity_qualification.py")
_FORBIDDEN_FRAGMENTS = (
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect.",
    "exec(",
    "eval(",
    "dict[str, Any]",
)


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


@pytest.fixture
def execution_identity_binding():
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    yield task_id, run_id, attempt_id
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_guardrail_clean_cross_system(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(eval_client=qualification_eval_client())
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-guardrail-pass",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_guardrail=True),
        shadow=shadow,
        guardrail_scan=guardrail_scan_allowed(),
    )
    assert ParityVerificationCapability.DETERMINISTIC_GUARDRAIL in (
        result.decision_observation.capabilities
    )
    assert ParityVerificationCapability.DETERMINISTIC_GUARDRAIL in (
        result.critic_observation.capabilities
    )


@pytest.mark.asyncio
async def test_guardrail_fail_match(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(eval_client=qualification_eval_client())
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-guardrail-fail",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_guardrail=True),
        shadow=shadow,
        guardrail_scan=guardrail_scan_blocked(),
    )
    assert result.classification is DecisionCriticParityClassification.MATCH
    assert result.retirement_blocking is False
    assert (
        result.decision_observation.outcome.value == "challenged"
        and result.critic_observation.outcome.value == "challenged"
    )


@pytest.mark.asyncio
async def test_semantic_pass_cross_system(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(
        eval_client=qualification_eval_client(),
        semantic=True,
    )
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-semantic-pass",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_semantic=True),
        shadow=shadow,
    )
    assert ParityVerificationCapability.SEMANTIC in result.decision_observation.capabilities
    assert ParityVerificationCapability.SEMANTIC in result.critic_observation.capabilities
    assert result.decision_observation.outcome.value == "acceptable"
    assert result.critic_observation.outcome.value == "acceptable"


@pytest.mark.asyncio
async def test_semantic_fail_match(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(
        eval_client=qualification_eval_client(),
        semantic=True,
    )
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-semantic-fail",
        summary=f"contains {SEMANTIC_FAIL_MARKER}",
        pipeline_options=QualificationPipelineOptions(include_semantic=True),
        shadow=shadow,
    )
    assert result.classification in {
        DecisionCriticParityClassification.MATCH,
        DecisionCriticParityClassification.EXPECTED_DIFFERENCE,
    }
    assert result.retirement_blocking is False
    assert result.decision_observation.outcome.value == "challenged"
    assert result.critic_observation.outcome.value == "challenged"


@pytest.mark.asyncio
async def test_semantic_shadow_unavailable_does_not_qualify(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    result = await run_semantic_shadow_unavailable_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-semantic-unavailable",
    )
    assert result.classification is DecisionCriticParityClassification.SHADOW_UNAVAILABLE


@pytest.mark.asyncio
async def test_semantic_required_judge_unavailable_challenges(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(
        eval_client=qualification_eval_client(),
        semantic=True,
    )
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-semantic-required-unavailable",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(
            include_semantic=True,
            semantic_judge=DeterministicSemanticJudge(available=False),
        ),
        shadow=shadow,
    )
    assert result.decision_observation.outcome.value == "challenged"
    assert result.decision_observation.verification_disposition is not None
    assert result.decision_observation.verification_disposition.value == "challenged"


@pytest.mark.asyncio
async def test_trajectory_pass_cross_system(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(
        eval_client=qualification_eval_client(),
        semantic=True,
        trajectory=True,
    )
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-trajectory-pass",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_trajectory=True),
        shadow=shadow,
    )
    assert ParityVerificationCapability.TRAJECTORY in result.decision_observation.capabilities
    assert ParityVerificationCapability.TRAJECTORY in result.critic_observation.capabilities


@pytest.mark.asyncio
async def test_trajectory_fail_match(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    fail_tenant_id = f"{TRAJECTORY_FAIL_MARKER}-tenant"
    shadow = build_qualification_shadow(
        eval_client=qualification_eval_client(),
        semantic=True,
        trajectory=True,
    )
    result = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        trajectory_tenant_id=fail_tenant_id,
        subject="graph-trajectory-fail",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_trajectory=True),
        shadow=shadow,
    )
    assert result.classification in {
        DecisionCriticParityClassification.MATCH,
        DecisionCriticParityClassification.EXPECTED_DIFFERENCE,
    }
    assert result.retirement_blocking is False


@pytest.mark.asyncio
async def test_evidence_decision_superset_pass_and_fail(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(eval_client=qualification_eval_client())
    valid = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-evidence-pass",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(
            include_evidence=True,
            evidence_valid=True,
        ),
        shadow=shadow,
        evidence_ref=str(KNOWN_EVIDENCE_REF),
    )
    invalid = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-evidence-fail",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(
            include_evidence=True,
            evidence_valid=False,
        ),
        shadow=shadow,
        evidence_ref=str(KNOWN_EVIDENCE_REF),
    )
    assert ParityVerificationCapability.EVIDENCE in valid.decision_observation.capabilities
    assert ParityVerificationCapability.EVIDENCE not in valid.critic_observation.capabilities
    assert valid.decision_observation.outcome.value == "acceptable"
    assert invalid.decision_observation.outcome.value == "challenged"


@pytest.mark.asyncio
async def test_domain_decision_superset_pass_and_fail(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(eval_client=qualification_eval_client())
    valid = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-domain-pass",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(include_domain=True),
        shadow=shadow,
    )
    invalid = await run_graph_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-domain-fail",
        summary=f"{DOMAIN_INVALID_MARKER} summary",
        pipeline_options=QualificationPipelineOptions(include_domain=True),
        shadow=shadow,
    )
    assert ParityVerificationCapability.DOMAIN in valid.decision_observation.capabilities
    assert ParityVerificationCapability.DOMAIN not in valid.critic_observation.capabilities
    assert valid.decision_observation.outcome.value == "acceptable"
    assert invalid.decision_observation.outcome.value == "challenged"


@pytest.mark.asyncio
async def test_hitl_architectural_mapping(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    result = await run_hitl_architectural_mapping_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        subject="graph-hitl",
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    assert result.decision_observation.human_review_pending is True
    assert LEGACY_HITL_IS_DECISION_HUMAN_REVIEW in {
        item.code for item in result.differences
    }


@pytest.mark.asyncio
async def test_uaep_scope_structural_observation(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    shadow = build_qualification_shadow(eval_client=qualification_eval_client())
    result = await run_uaep_parity_case(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
        step_id="step-1",
        summary="valid summary",
        pipeline_options=QualificationPipelineOptions(),
        shadow=shadow,
    )
    assert result.identity.host_scope is ParityHostScope.UAEP_STEP
    assert ParityVerificationCapability.STRUCTURAL in result.decision_observation.capabilities


async def _collect_qualification_results(
    *,
    task_id,
    run_id,
    attempt_id,
):
    eval_client = qualification_eval_client()
    shadow = build_qualification_shadow(eval_client=eval_client)
    semantic_shadow = build_qualification_shadow(eval_client=eval_client, semantic=True)
    full_shadow = build_qualification_shadow(
        eval_client=eval_client,
        semantic=True,
        trajectory=True,
    )

    return (
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-structural",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(),
            shadow=shadow,
        ),
        await run_uaep_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            step_id="step-1",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(),
            shadow=shadow,
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-guardrail-pass",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_guardrail=True),
            shadow=shadow,
            guardrail_scan=guardrail_scan_allowed(),
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-guardrail-fail",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_guardrail=True),
            shadow=shadow,
            guardrail_scan=guardrail_scan_blocked(),
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-semantic-pass",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_semantic=True),
            shadow=semantic_shadow,
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-semantic-fail",
            summary=f"contains {SEMANTIC_FAIL_MARKER}",
            pipeline_options=QualificationPipelineOptions(include_semantic=True),
            shadow=semantic_shadow,
        ),
        await run_semantic_shadow_unavailable_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-semantic-unavailable",
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-trajectory-pass",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_trajectory=True),
            shadow=full_shadow,
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            trajectory_tenant_id=f"{TRAJECTORY_FAIL_MARKER}-tenant",
            subject="graph-trajectory-fail",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_trajectory=True),
            shadow=full_shadow,
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-evidence-pass",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(
                include_evidence=True,
                evidence_valid=True,
            ),
            shadow=shadow,
            evidence_ref=str(KNOWN_EVIDENCE_REF),
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-evidence-fail",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(
                include_evidence=True,
                evidence_valid=False,
            ),
            shadow=shadow,
            evidence_ref=str(KNOWN_EVIDENCE_REF),
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-domain-pass",
            summary="valid summary",
            pipeline_options=QualificationPipelineOptions(include_domain=True),
            shadow=shadow,
        ),
        await run_graph_parity_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-domain-fail",
            summary=f"{DOMAIN_INVALID_MARKER} summary",
            pipeline_options=QualificationPipelineOptions(include_domain=True),
            shadow=shadow,
        ),
        await run_hitl_architectural_mapping_case(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
            subject="graph-hitl",
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )


@pytest.mark.asyncio
async def test_full_retirement_qualification_evidence(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    qualification_results = await _collect_qualification_results(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    report = evaluate_critic_retirement_readiness(
        qualification_results,
        required_scopes=frozenset({
            ParityHostScope.GRAPH_FINAL,
            ParityHostScope.UAEP_STEP,
        }),
        capability_requirements=DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    )
    metrics = aggregate_parity_metrics(qualification_results)
    assert metrics.total_comparisons == len(qualification_results)
    assert report.readiness is CriticRetirementReadiness.READY
    assert report.missing_capabilities == frozenset()
    assert report.missing_scopes == frozenset()
    assert report.blocking_mismatch_count == 0
    assert report.shadow_error_count == 0
    assert report.shadow_unavailable_count == 1
    assert metrics.retirement_blocking_mismatches == 0
    assert metrics.shadow_errors == 0
    assert report.scopes_exercised == frozenset({
        ParityHostScope.GRAPH_FINAL,
        ParityHostScope.UAEP_STEP,
    })
    assert report.cross_system_capabilities_qualified == frozenset({
        ParityVerificationCapability.STRUCTURAL,
        ParityVerificationCapability.DETERMINISTIC_GUARDRAIL,
        ParityVerificationCapability.SEMANTIC,
        ParityVerificationCapability.TRAJECTORY,
    })
    assert report.decision_superset_capabilities_qualified == frozenset({
        ParityVerificationCapability.EVIDENCE,
        ParityVerificationCapability.DOMAIN,
    })
    assert report.architectural_mappings_qualified == frozenset({
        ParityVerificationCapability.HUMAN_HITL,
    })
    decision_superset_cases = tuple(
        result
        for result in qualification_results
        if result.identity.subject in {"graph-evidence-fail", "graph-domain-fail"}
    )
    assert len(decision_superset_cases) == 2
    for result in decision_superset_cases:
        assert result.classification is DecisionCriticParityClassification.MISMATCH
        assert result.retirement_blocking is False
        assert DECISION_SUPERSET_CAPABILITY in {item.code for item in result.differences}


def test_forbidden_audit_qualification_support() -> None:
    source = _SUPPORT_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source


def test_parity_qualification_case_contract() -> None:
    case = ParityQualificationCase(
        case_id="semantic-pass",
        scope=ParityHostScope.GRAPH_FINAL,
        capability=ParityVerificationCapability.SEMANTIC,
        expected_classification=DecisionCriticParityClassification.MATCH,
        mode=ParityQualificationMode.CROSS_SYSTEM,
    )
    assert case.case_id == "semantic-pass"
