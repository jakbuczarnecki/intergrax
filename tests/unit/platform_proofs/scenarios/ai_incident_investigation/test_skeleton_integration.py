# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from intergrax.runtime.critic.critic_wiring import (
    CriticHookConfig,
    build_critic_graph_hooks,
)
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime, ToolInvocationPlan
from scripts.proof.intergrax_platform_proof_evidence import (
    iter_evidence_claim_graph_binding_violations,
    PlatformProofEvidence,
)
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report
from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures import FORBIDDEN_LEAK_MARKERS
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INVESTIGATOR_CAPABILITY,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    build_runtime_bundle,
    execute_resolved_skeleton,
    OUTCOME_RESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    register_scenario_tools,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    IncidentInvestigationValidationEngine,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_resolved_skeleton_executes_platform_path() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_challenged
    assert result.evaluator_loop_iterations >= 1
    assert result.tool_invocations >= 3
    assert result.revision_used_tools

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    supported = [c for c in claim_set.claims if c.resolution is ClaimResolution.SUPPORTED]
    assert supported
    assert TELEMETRY_EVIDENCE_ID in supported[-1].supporting_evidence_ids

    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed


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
async def test_critic_completion_gate_blocks_when_l1_required() -> None:
    from intergrax.contracts.agent_execution_result import (
        AgentExecutionResult,
        AgentExecutionStatus,
    )
    from intergrax.contracts.execution_identity import mint_run_id
    from intergrax.runtime.critic.critic_wiring import (
        CriticHookConfig,
        build_critic_graph_hooks,
        validate_final_with_critic_detail,
    )

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


@pytest.mark.asyncio
async def test_platform_proof_evidence_verifier_and_renderer() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(result, source_revision="testsha")
    graph_ids = frozenset(node.evidence_id for node in evidence.evidence_graph.nodes)
    violations = iter_evidence_claim_graph_binding_violations(
        evidence.evidence_claims,
        graph_ids,
    )
    assert not violations
    PlatformProofEvidence.model_validate(evidence.model_dump(mode="json"))

    html = render_platform_proof_report(evidence)
    lowered = html.lower()
    assert "material" in lowered and "claim" in lowered
    assert "station signal" in lowered or "equipment" in lowered


@pytest.mark.asyncio
async def test_tampered_evidence_reference_fails_verification() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(result, source_revision="testsha")
    tampered = json.loads(evidence.model_dump_json())
    tampered["evidence_claims"]["claims"][0]["supporting_evidence_ids"] = [
        "evidence.tampered.missing"
    ]
    with pytest.raises(ValueError, match="claim_support_evidence_missing"):
        PlatformProofEvidence.model_validate(tampered)


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

