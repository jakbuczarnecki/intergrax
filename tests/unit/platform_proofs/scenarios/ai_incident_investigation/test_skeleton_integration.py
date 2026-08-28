# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
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
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
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
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    load_manifest_bundle,
)
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report
from scripts.proof.intergrax_proof_contracts import (
    EvidenceVerificationStatus as ContractEvidenceStatus,
    ProofStatus,
)
from scripts.proof.intergrax_proof_runner import execute_proof, read_git_metadata
from platform_proofs.scenarios.ai_incident_investigation.application.critic_adapter import (
    apply_challenge_lifecycle,
    build_satisfied_challenge,
    map_critic_verdict_to_challenge,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evidence_builder import (
    EVIDENCE_RESOLVED_FILENAME,
    PROOF_ID,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    FORBIDDEN_LEAK_MARKERS,
    ScenarioVariant,
    TimeWindowLabel,
    build_resolved_fixture,
    staffing_record_admissible_for_incident,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    INITIAL_CLAIM_ID,
    INVESTIGATOR_CAPABILITY,
    REVISED_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    EVALUATOR_LOOP_MAX_ITERATIONS,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
    execute_with_completion_gate_blocked,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    register_scenario_tools,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
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


def _failed_critic_verdict() -> CriticVerdict:
    return CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                errors=[UNSUPPORTED_INFERENCE_ERROR],
            )
        ],
        recommended_action=CriticAction.REVISE,
        failure_reasons=[UNSUPPORTED_INFERENCE_ERROR],
    )


def _empty_claim_set() -> dict[str, object]:
    return {
        "schema_version": "evidence_claim_set.v1",
        "claims": [
            {
                "claim_id": str(INITIAL_CLAIM_ID),
                "statement": "initial overload diagnosis",
                "claim_kind": "incident.root_cause_diagnosis",
                "supporting_evidence_ids": [
                    str(WORKLOAD_EVIDENCE_ID),
                    str(THROUGHPUT_EVIDENCE_ID),
                ],
                "resolution": "pending",
            }
        ],
        "challenges": [],
    }


def _skeleton_manifest_entry(repo_root: Path):
    bundle = load_manifest_bundle(repo_root=repo_root)
    entry = next(
        item for item in bundle.manifest.entries if item.proof_id == PROOF_ID
    )
    return entry, bundle.execution_specs[PROOF_ID]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


@pytest.mark.asyncio
async def test_resolved_skeleton_executes_platform_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_challenged
    assert result.evaluator_loop_iterations >= 1
    assert result.evaluator_loop_iterations <= EVALUATOR_LOOP_MAX_ITERATIONS
    assert result.tool_invocations >= 6
    assert result.revision_used_tools
    assert result.failed_critic_verdict is not None
    assert UNSUPPORTED_INFERENCE_ERROR in result.failed_critic_verdict.failure_reasons

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    supported = [c for c in claim_set.claims if c.resolution is ClaimResolution.SUPPORTED]
    assert supported
    assert TELEMETRY_EVIDENCE_ID in supported[-1].supporting_evidence_ids
    assert result.evidence_challenge is not None
    assert result.evidence_challenge.resolution is ChallengeResolution.SATISFIED
    assert TELEMETRY_EVIDENCE_ID in result.evidence_challenge.evidence_ids
    assert WORKLOAD_EVIDENCE_ID in result.evidence_challenge.evidence_ids
    assert COMPARISON_EVIDENCE_ID in result.evidence_challenge.evidence_ids

    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed


@pytest.mark.asyncio
async def test_real_critic_provenance_maps_to_challenge_with_stable_id() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    assert result.failed_critic_verdict is not None
    assert UNSUPPORTED_INFERENCE_ERROR in result.failed_critic_verdict.failure_reasons
    assert result.evidence_challenge is not None
    assert result.challenged_claim_id is not None
    assert result.evidence_challenge.claim_id == result.challenged_claim_id
    open_challenge = map_critic_verdict_to_challenge(
        result.failed_critic_verdict,
        claim_id=result.evidence_challenge.claim_id,
    )
    assert open_challenge is not None
    assert UNSUPPORTED_INFERENCE_ERROR in open_challenge.description
    assert result.evidence_challenge.resolution is ChallengeResolution.SATISFIED

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    assert len(claim_set.challenges) == 1
    assert claim_set.challenges[0].challenge_id == result.evidence_challenge.challenge_id


def test_apply_challenge_lifecycle_open_excludes_resolving_evidence() -> None:
    failed_verdict = _failed_critic_verdict()
    _, open_challenge = apply_challenge_lifecycle(
        _empty_claim_set(),
        failed_verdict,
        claim_id=INITIAL_CLAIM_ID,
        initial_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
        resolving_evidence_ids=(TELEMETRY_EVIDENCE_ID,),
        resolved=False,
    )
    assert open_challenge is not None
    assert TELEMETRY_EVIDENCE_ID not in open_challenge.evidence_ids
    assert WORKLOAD_EVIDENCE_ID in open_challenge.evidence_ids
    assert THROUGHPUT_EVIDENCE_ID in open_challenge.evidence_ids


def test_apply_challenge_lifecycle_satisfied_includes_resolving_evidence() -> None:
    failed_verdict = _failed_critic_verdict()
    satisfied_set, satisfied_challenge = apply_challenge_lifecycle(
        _empty_claim_set(),
        failed_verdict,
        claim_id=INITIAL_CLAIM_ID,
        initial_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
        resolving_evidence_ids=(TELEMETRY_EVIDENCE_ID,),
        resolved=True,
    )
    assert satisfied_challenge is not None
    assert satisfied_challenge.resolution is ChallengeResolution.SATISFIED
    assert TELEMETRY_EVIDENCE_ID in satisfied_challenge.evidence_ids
    claim_set = EvidenceClaimSet.model_validate(satisfied_set)
    assert len(claim_set.challenges) == 1


def test_build_satisfied_challenge_preserves_open_challenge_id() -> None:
    failed_verdict = _failed_critic_verdict()
    open_challenge = map_critic_verdict_to_challenge(
        failed_verdict,
        claim_id=INITIAL_CLAIM_ID,
        evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
    )
    assert open_challenge is not None
    satisfied = build_satisfied_challenge(
        open_challenge.challenge_id,
        claim_id=INITIAL_CLAIM_ID,
        evidence_ids=(
            WORKLOAD_EVIDENCE_ID,
            THROUGHPUT_EVIDENCE_ID,
            TELEMETRY_EVIDENCE_ID,
        ),
        description="Follow-up telemetry gathered via platform tools",
    )
    assert satisfied.challenge_id == open_challenge.challenge_id
    assert TELEMETRY_EVIDENCE_ID in satisfied.evidence_ids


@pytest.mark.asyncio
async def test_completion_gate_required_on_resolved_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed


@pytest.mark.asyncio
async def test_completion_gate_blocks_resolved_on_real_scenario_path() -> None:
    bundle = build_runtime_bundle()
    with pytest.raises(RuntimeError, match="incident_terminal_state_not_accepted"):
        await execute_with_completion_gate_blocked(bundle)


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
                            "resolution": "pending",
                        }
                    ],
                    "challenges": [],
                },
                "claim_hypothesis_bindings": [
                    {"claim_id": str(REVISED_CLAIM_ID), "hypothesis_id": "H3"},
                ],
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
    assert (
        "supported_diagnosis_telemetry_not_observable" in validation.errors
        or "supported_diagnosis_evidence_not_observable" in validation.errors
        or "h3_diagnosis_telemetry_not_observable" in validation.errors
        or "unsupported_inference:missing_comparison_evidence" in validation.errors
    )


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
    assert result.tool_invocations == 6


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
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    evidence = build_platform_proof_evidence(
        result,
        evaluation=evaluation,
        variant=ScenarioVariant.RESOLVED,
        source_revision="testsha",
    )
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
        assert str(TELEMETRY_EVIDENCE_ID) in projected.challenges[0].evidence_ids

    PlatformProofEvidence.model_validate(evidence.model_dump(mode="json"))
    html = render_platform_proof_report(evidence)
    lowered = html.lower()
    assert "material" in lowered and "claim" in lowered
    assert "station signal" in lowered or "equipment" in lowered


@pytest.mark.asyncio
async def test_parent_runner_integration_valid_scenario_skeleton(
    repo_root: Path,
) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    assert entry.public_evidence_eligible is False
    git = read_git_metadata(repo_root)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
        )
        assert result.exit_code == 0
        assert result.status == ProofStatus.PASS
        assert result.evidence_verification_status == ContractEvidenceStatus.PASS
        assert (artifact_dir / EVIDENCE_RESOLVED_FILENAME).is_file()


@pytest.mark.asyncio
async def test_parent_runner_integration_tampered_evidence_fails(
    repo_root: Path,
) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    git = read_git_metadata(repo_root)

    def _tamper_runner(command, **kwargs):
        completed = subprocess.run(command, **kwargs)
        artifact_dir = Path(kwargs["env"][INTERGRAX_PROOF_ARTIFACT_DIR_ENV])
        evidence_path = artifact_dir / EVIDENCE_RESOLVED_FILENAME
        tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        tampered["evidence_claims"]["claims"][0]["supporting_evidence_ids"] = [
            "evidence.tampered.missing"
        ]
        evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
        return completed

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
            subprocess_runner=_tamper_runner,
        )
    assert result.status == ProofStatus.FAIL
    assert result.evidence_verification_status == ContractEvidenceStatus.INVALID
    assert result.diagnostic_summary in {
        "claim_support_evidence_missing",
        "invalid_evidence",
        "evidence failed PlatformProofEvidence validation",
    }


def test_static_audit_child_run_proof_has_no_self_verification() -> None:
    run_proof_path = (
        Path(__file__).resolve().parents[5]
        / "platform_proofs"
        / "scenarios"
        / "ai_incident_investigation"
        / "run_proof.py"
    )
    source = run_proof_path.read_text(encoding="utf-8")
    forbidden = (
        "ProofRunResult(",
        "verify_platform_proof_evidence(",
        "_skeleton_execution_spec",
        "_verify_written_evidence",
    )
    for token in forbidden:
        assert token not in source


@pytest.mark.asyncio
async def test_critic_hooks_wired_for_investigator() -> None:
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True, verify_graph_final=True),
        validation_engine=IncidentInvestigationValidationEngine(),
    )
    assert hooks is not None


def test_static_bypass_audit_no_substitute_runtimes() -> None:
    package = (
        Path(__file__).resolve().parents[5]
        / "platform_proofs"
        / "scenarios"
        / "ai_incident_investigation"
    )
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
    sources = [p.read_text(encoding="utf-8") for p in package.glob("**/*.py")]
    combined = "\n".join(sources)
    for name in forbidden_names:
        assert f"class {name}" not in combined
    assert "CriticVerdict(" not in combined or "first_failed_node_partial_verdict_from_trace" in combined
    assert 'summary.startswith("revised:")' not in combined
    assert 'summary.startswith("draft:")' not in combined
