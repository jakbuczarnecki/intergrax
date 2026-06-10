# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL depth gate evidence (Band 2az)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.async_task_index_resolver import resolve_async_task_index
from intergrax.applications._shared.intake_wiring import resolve_product_intake_wiring
from intergrax.applications._shared.modality_production_resolver import resolve_live_vision_profile
from intergrax.applications._shared.tenant_storage_wiring import (
    resolve_tenant_postgresql_config,
    tenant_storage_isolation_ready,
)
from intergrax.applications._shared.compensation_wiring import resolve_compensation_flow
from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    resolve_production_queue_backend,
)
from intergrax.applications._shared.reasoning_wiring import resolve_replan_policy_context
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications._shared.sandbox_wiring import product_requires_sandbox
from intergrax.applications._shared.sqlite_async_task_index import SqliteAsyncTaskIndex
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.rag.profiles.rag_profile import production_rag_profile
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.adaptive.l4_runtime_evidence import build_harness_baseline_l4_evidence
from intergrax.runtime.architecture.release_cycle_tracker import resolve_release_cycle_count
from intergrax.runtime.context.context_drift_monitor import ContextDriftSignal, evaluate_context_drift
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.applications._shared.registry_snapshot_store import persist_registry_snapshot
from intergrax.applications._shared.replay_routes import create_replay_router
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.memory.org_memory_scope import OrgMemoryScope
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_output_policy_bridge import apply_pre_output_policy
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.gate

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_audit_ideal_3_1_envelope_runtime_roundtrip() -> None:
    envelope = TaskEnvelope(tenant_id="t1", user_id="u1", message="hi", agent_id="echo")
    request = RuntimeRequest.from_envelope(envelope)
    assert request.tenant_id == "t1"
    assert request.message == "hi"
    assert Task.from_envelope(envelope).to_envelope().message == "hi"


def test_audit_ideal_5_1_pre_output_policy() -> None:
    engine = PolicyEngine()
    task = Task(tenant_id="t1", user_id="u1", agent_id="echo", message="m")
    answer, decision = apply_pre_output_policy(engine, task, answer="valid answer")
    assert answer == "valid answer"
    assert decision.action.value == "allow"
    blocked, _ = apply_pre_output_policy(engine, task, answer="")
    assert blocked.startswith("[POLICY_BLOCKED]")


def test_audit_ideal_7_1_reasoning_profile_exists() -> None:
    profile = ReasoningProfile()
    assert profile.planner_prompt_id == "nexus_task_planner"


def test_audit_ideal_15_1_org_memory_scopes() -> None:
    assert OrgMemoryScope.ORG_PROFILE.value == "org_profile"
    assert len(list(OrgMemoryScope)) == 3


def test_audit_ideal_19_1_registry_snapshot_store(tmp_path: Path) -> None:
    snapshot = HarnessRegistrySnapshot(
        integration_profile=None,
        tool_registry=None,
        skill_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )
    sid = persist_registry_snapshot(
        snapshot, host_id="lab", db_path=tmp_path / "registry.db", snapshot_id="snap_test"
    )
    assert sid == "snap_test"


def test_audit_ideal_27_2_replay_router() -> None:
    router = create_replay_router(enabled=True)
    assert any(route.path.endswith("/replay") for route in router.routes)


def test_audit_ideal_30_1_ecp_architecture_synced() -> None:
    arch = REPO_ROOT / "docs" / "architecture" / "ELASTIC_CAPACITY_AND_SCALING.md"
    text = arch.read_text(encoding="utf-8")
    assert "Harness elastic control loop" in text
    assert "L3" in text


def test_audit_ideal_7_2_replan_policy_context() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": ApplicationEnvironmentProfile.lab_defaults()
            .orchestration_profile.model_copy(update={"allow_dynamic_replan": True})
        }
    )
    ctx = resolve_replan_policy_context(env)
    assert ctx.get("engine_replan_boundary") is True
    handler = ExecutionInterruptHandler(allow_dynamic_replan=True)
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan"),
        task_id="t",
        run_id="r",
        agent_id="echo",
        context=ctx,
    )
    assert resolution.policy_decision.action is PolicyAction.ALLOW


def test_audit_ideal_9_1_production_queue_backend() -> None:
    assert resolve_production_queue_backend(env_value="celery") is ProductionQueueBackend.CELERY


def test_audit_ideal_11_1_sandbox_product_requirement() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["sandbox.exec"])
    assert product_requires_sandbox(env) is True


def test_audit_ideal_14_1_graph_rag_production_profile() -> None:
    profile = production_rag_profile()
    assert profile.graph_rag_enabled is True


def test_audit_ideal_22_1_compensation_flow() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    flow = resolve_compensation_flow(env)
    assert flow is not None
    assert flow.handlers


def test_audit_ideal_28_1_durable_async_index(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    index = resolve_async_task_index(env, db_path=tmp_path / "audit_ideal_async.db")
    assert isinstance(index, SqliteAsyncTaskIndex)


def test_audit_ideal_31_2_promotion_requires_eval() -> None:
    bundle = PromotionEvidenceBundle(
        agent_id="echo",
        agent_version="1.0.0",
        source_stage=PromotionStage.STAGING,
        target_stage=PromotionStage.PRODUCTION,
        certification=AgentCertificationEvaluation(
            agent_id="echo",
            agent_version="1.0.0",
            eligible=True,
            reasons=[],
        ),
        evaluation_report_refs=[],
        rollback_plan_ref="rb",
        change_ticket_ref="chg",
    )
    assert evaluate_agent_promotion(bundle).approved is False


def test_audit_ideal_ahi_1_l4_baseline_evidence() -> None:
    report = build_harness_baseline_l4_evidence()
    assert report.runtime_l4_closed_loop_passed is True
    assert report.scenarios_passed_count >= 3


def test_audit_ideal_4_2_tenant_storage_isolation() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    assert tenant_storage_isolation_ready(env)
    config = resolve_tenant_postgresql_config("acme")
    assert config.tenant_schema == "acme"


def test_audit_ideal_16_1_context_drift_monitor() -> None:
    report = evaluate_context_drift(
        ContextDriftSignal(token_estimate=1500, chunk_count=1, baseline_token_estimate=1000),
    )
    assert report.alert is True


def test_audit_ideal_29_1_live_vision_profile() -> None:
    profile = resolve_live_vision_profile()
    assert profile.create_adapter().slug


def test_audit_ideal_30_2_deploy_slo_evidence() -> None:
    assert resolve_release_cycle_count(repo_root=REPO_ROOT) >= 2


def test_audit_ideal_3_2_product_intake_parity() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_product_intake_wiring(env)
    assert wiring.durable_async_index is True
    assert wiring.streaming_intake_enabled is True


def test_audit_ideal_deferred_register() -> None:
    register = REPO_ROOT / "docs" / "plan" / "AUDIT_IDEAL_2026.md"
    text = register.read_text(encoding="utf-8")
    for task_id in ("28.3", "28.4", "5.3", "21.3"):
        assert f"AUDIT-IDEAL-{task_id}" in text
        assert "Deferred" in text
