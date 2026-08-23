# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.evidence_claims import (
    ChallengeResolution,
    ClaimResolution,
    EvidenceClaimSet,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.critic.critic_wiring import (
    CriticHookConfig,
    build_critic_graph_hooks,
    validate_final_with_critic_detail,
)
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime, ToolInvocationPlan
from scripts.proof.intergrax_platform_proof_evidence import (
    PlatformProofEvidence,
    ReportSafeTextSourceKind,
    project_evidence_claim_set,
)
from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    EvidenceVerificationStatus,
    verify_platform_proof_evidence,
)
from scripts.proof.intergrax_platform_proof_execution import ProofExecutionSpec
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report
from scripts.proof.intergrax_proof_contracts import (
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.critic_adapter import (
    map_critic_verdict_to_challenge,
)
from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    PROOF_ID,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures import FORBIDDEN_LEAK_MARKERS
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INITIAL_CLAIM_ID,
    INVESTIGATOR_CAPABILITY,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    EVALUATOR_LOOP_MAX_ITERATIONS,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
    execute_with_completion_gate_blocked,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    register_scenario_tools,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    IncidentInvestigationValidationEngine,
    UNSUPPORTED_INFERENCE_ERROR,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def _skeleton_execution_spec() -> ProofExecutionSpec:
    return ProofExecutionSpec(
        manifest_entry=ProofManifestEntry(
            proof_id=PROOF_ID,
            title="AI Incident Investigation — platform-native skeleton",
            profiles=frozenset({ProofProfile.QUICK}),
            proof_kind="scenario_skeleton",
            command=ProofArgvCommand(
                executable="python",
                argv=("platform_proofs/scenarios/ai_incident_investigation/run_proof.py",),
            ),
            safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
        ),
        evidence_required=True,
    )


@pytest.mark.asyncio
async def test_resolved_skeleton_executes_platform_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_challenged
    assert result.evaluator_loop_iterations >= 1
    assert result.evaluator_loop_iterations <= EVALUATOR_LOOP_MAX_ITERATIONS
    assert result.tool_invocations >= 3
    assert result.revision_used_tools
    assert result.failed_critic_verdict is not None
    assert UNSUPPORTED_INFERENCE_ERROR in result.failed_critic_verdict.failure_reasons

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    supported = [c for c in claim_set.claims if c.resolution is ClaimResolution.SUPPORTED]
    assert supported
    assert TELEMETRY_EVIDENCE_ID in supported[-1].supporting_evidence_ids
    assert result.evidence_challenge is not None
    assert result.evidence_challenge.resolution is ChallengeResolution.SATISFIED

    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed


@pytest.mark.asyncio
async def test_real_critic_provenance_maps_to_challenge_with_stable_id() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    assert result.failed_critic_verdict is not None
    assert UNSUPPORTED_INFERENCE_ERROR in result.failed_critic_verdict.failure_reasons
    assert result.evidence_challenge is not None
    assert result.evidence_challenge.claim_id == INITIAL_CLAIM_ID
    open_challenge = map_critic_verdict_to_challenge(
        result.failed_critic_verdict,
        claim_id=INITIAL_CLAIM_ID,
    )
    assert open_challenge is not None
    assert UNSUPPORTED_INFERENCE_ERROR in open_challenge.description
    assert result.evidence_challenge.resolution is ChallengeResolution.SATISFIED

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    assert len(claim_set.challenges) == 1
    assert claim_set.challenges[0].challenge_id == result.evidence_challenge.challenge_id


@pytest.mark.asyncio
async def test_completion_gate_required_on_resolved_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed


@pytest.mark.asyncio
async def test_completion_gate_blocks_resolved_on_real_scenario_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_with_completion_gate_blocked(bundle)
    assert result.outcome == OUTCOME_UNRESOLVED
    assert not result.critic_verdict_passed


@pytest.mark.asyncio
async def test_summary_revised_prefix_does_not_bypass_validation() -> None:
    engine = IncidentInvestigationValidationEngine()
    execution = AgentExecutionResult(
        agent_id="incident_investigator",
        run_id=mint_run_id(),
        status=AgentExecutionStatus.COMPLETED,
        summary="revised: bounded equipment-process degradation diagnosis supported by telemetry",
        structured_data={
            "domain_summary": {
                "claim_set": {
                    "schema_version": "evidence_claim_set.v1",
                    "claims": [],
                    "challenges": [],
                },
                "evidence_nodes": [],
                "active_hypothesis": "H3",
            }
        },
    )
    validation = engine.validate(
        execution,
        contract=build_runtime_bundle().investigator.get_contract(),
        capability=INVESTIGATOR_CAPABILITY,
    )
    assert not validation.valid
    assert "missing_diagnosis_claim" in validation.errors


@pytest.mark.asyncio
async def test_summary_revised_with_missing_telemetry_evidence_fails() -> None:
    engine = IncidentInvestigationValidationEngine()
    execution = AgentExecutionResult(
        agent_id="incident_investigator",
        run_id=mint_run_id(),
        status=AgentExecutionStatus.COMPLETED,
        summary="revised: bounded equipment-process degradation diagnosis supported by telemetry",
        structured_data={
            "domain_summary": {
                "claim_set": {
                    "schema_version": "evidence_claim_set.v1",
                    "claims": [
                        {
                            "claim_id": str(REVISED_CLAIM_ID),
                            "statement": "bounded H3 diagnosis",
                            "claim_kind": "incident.root_cause_diagnosis",
                            "supporting_evidence_ids": [
                            "evidence.telemetry.fake.not_in_graph"
                            ],
                            "resolution": "supported",
                        }
                    ],
                    "challenges": [],
                },
                "evidence_nodes": [],
                "active_hypothesis": "H3",
            }
        },
    )
    validation = engine.validate(
        execution,
        contract=build_runtime_bundle().investigator.get_contract(),
        capability=INVESTIGATOR_CAPABILITY,
    )
    assert not validation.valid
    assert "supported_diagnosis_telemetry_not_observable" in validation.errors


@pytest.mark.asyncio
async def test_tool_runtime_invoked_not_mocked_away() -> None:
    bundle = build_runtime_bundle()
    registry = ToolRegistry()
    register_scenario_tools(registry, bundle.fixture)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tool_invoker=invoker,
        tools_mode="catalog",
    )
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext

    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=pytest.importorskip("unittest.mock").MagicMock(),
    )
    run_id = mint_run_id()
    state = RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="probe",
            user_id="u",
            session_id="s",
            tenant_id="t",
            task_id=mint_task_id(),
            run_id=run_id,
            message="probe",
        ),
        run_id=run_id,
        tool_traces=[],
    )
    plan = ToolInvocationPlan(
        tool_ids=(TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ),
        tool_inputs={
            TOOL_WORKLOAD_READ: {"line_id": "line4", "window": "incident_window"},
            TOOL_THROUGHPUT_READ: {"line_id": "line4", "window": "incident_window"},
        },
    )
    runtime_result = await ToolRuntime.invoke(state=state, plan=plan)
    assert runtime_result.tool_trace_count >= 2
    assert len(state.tool_traces) >= 2


@pytest.mark.asyncio
async def test_follow_up_uses_platform_tools_not_direct_fixture() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.revision_used_tools
    assert result.tool_invocations == 3


@pytest.mark.asyncio
async def test_ground_truth_not_in_model_visible_blob() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    blob = result.leak_scan_blob.lower()
    for marker in FORBIDDEN_LEAK_MARKERS:
        assert marker.lower() not in blob


@pytest.mark.asyncio
async def test_critic_completion_gate_blocks_when_l1_required_synthetic() -> None:
    bundle = build_runtime_bundle()
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(
            verify_node_partial=True,
            verify_graph_final=True,
            require_critic_on_completion=True,
            semantic_judge_enabled=True,
        ),
        validation_engine=IncidentInvestigationValidationEngine(),
    )
    assert hooks is not None
    execution = AgentExecutionResult(
        agent_id=bundle.investigator.get_contract().id,
        run_id=mint_run_id(),
        status=AgentExecutionStatus.COMPLETED,
        summary="revised: bounded equipment-process degradation diagnosis supported by telemetry",
        structured_data={
            "domain_summary": {
                "claim_set": {
                    "schema_version": "evidence_claim_set.v1",
                    "claims": [],
                    "challenges": [],
                }
            }
        },
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    validation, verdict = validate_final_with_critic_detail(
        execution,
        contract=bundle.investigator.get_contract(),
        hooks=hooks,
        task_id=task_id,
        run_id=run_id,
        tenant_id="scenario-tenant",
        capability=INVESTIGATOR_CAPABILITY,
    )
    assert not validation.valid
    assert "critic_completion_blocked" in validation.errors
    assert not verdict.passed or "critic_completion_blocked" in validation.errors


@pytest.mark.asyncio
async def test_platform_proof_evidence_verifier_and_renderer() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(result, source_revision="testsha")
    projected = project_evidence_claim_set(
        EvidenceClaimSet.model_validate(result.claim_set),
        text_source=ReportSafeTextSourceKind.RUNTIME_EXPLICIT,
    )
    assert projected.claims[0].statement.source_kind is ReportSafeTextSourceKind.RUNTIME_EXPLICIT
    if projected.challenges:
        assert (
            projected.challenges[0].description.source_kind
            is not ReportSafeTextSourceKind.PROOF_AUTHORED
        )
        assert projected.challenges[0].description.source_kind in {
            ReportSafeTextSourceKind.RUNTIME_EXPLICIT,
            ReportSafeTextSourceKind.RUNTIME_SANITIZED,
        }

    PlatformProofEvidence.model_validate(evidence.model_dump(mode="json"))
    html = render_platform_proof_report(evidence)
    lowered = html.lower()
    assert "material" in lowered and "claim" in lowered
    assert "station signal" in lowered or "equipment" in lowered


@pytest.mark.asyncio
async def test_canonical_verifier_passes_valid_artifact() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(result, source_revision="testsha")
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        evidence_path = write_evidence_json(evidence, proof_directory=artifact_dir)
        transport = ProofRunResult(
            proof_id=PROOF_ID,
            status=ProofStatus.PASS,
            exit_code=0,
            duration_seconds=0.0,
        )
        verification = verify_platform_proof_evidence(
            evidence_path=evidence_path,
            artifact_root=artifact_dir,
            spec=_skeleton_execution_spec(),
            subprocess_result=transport,
            expected_source_revision="testsha",
        )
        assert verification.status is EvidenceVerificationStatus.PASS, (
            verification.diagnostic_code,
            verification.diagnostic_summary,
        )


@pytest.mark.asyncio
async def test_canonical_verifier_rejects_tampered_evidence_reference() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(result, source_revision="testsha")
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        evidence_path = write_evidence_json(evidence, proof_directory=artifact_dir)
        tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        tampered["evidence_claims"]["claims"][0]["supporting_evidence_ids"] = [
            "evidence.tampered.missing"
        ]
        evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
        transport = ProofRunResult(
            proof_id=PROOF_ID,
            status=ProofStatus.PASS,
            exit_code=0,
            duration_seconds=0.0,
        )
        verification = verify_platform_proof_evidence(
            evidence_path=evidence_path,
            artifact_root=artifact_dir,
            spec=_skeleton_execution_spec(),
            subprocess_result=transport,
            expected_source_revision="testsha",
        )
        assert verification.status is EvidenceVerificationStatus.INVALID
        assert verification.diagnostic_code in {
            "claim_support_evidence_missing",
            "invalid_evidence",
        }


@pytest.mark.asyncio
async def test_critic_hooks_wired_for_investigator() -> None:
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True, verify_graph_final=True),
        validation_engine=IncidentInvestigationValidationEngine(),
    )
    assert hooks is not None


def test_static_bypass_audit_no_substitute_runtimes() -> None:
    package = Path(__file__).resolve().parents[4] / "platform_proofs" / "scenarios" / "ai_incident_investigation"
    forbidden_names = (
        "RetryLoop",
        "VerifierRuntime",
        "ToolDispatcher",
        "RunJournal",
        "EventBus",
        "EvidenceStore",
        "ClaimModel",
        "ReportRenderer",
        "AgentLoop",
    )
    sources = [p.read_text(encoding="utf-8") for p in package.glob("*.py")]
    combined = "\n".join(sources)
    for name in forbidden_names:
        assert f"class {name}" not in combined
    assert "CriticVerdict(" not in combined or "first_failed_node_partial_verdict_from_trace" in combined
    assert 'summary.startswith("revised:")' not in combined
    assert 'summary.startswith("draft:")' not in combined
