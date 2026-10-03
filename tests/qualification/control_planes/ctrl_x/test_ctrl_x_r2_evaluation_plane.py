# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R2 Evaluation authority proofs (R2-C)."""

from __future__ import annotations

import inspect

import pytest

from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.online_evaluation import record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationMode
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.decision_verification_composition import build_decision_verification_pipeline
from intergrax.runtime.decision_verification_composition import DecisionVerificationPipelineBuildSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_r2_eval_01_observation_path_appends_evidence_only() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    obs = record_shadow_observation(
        run_id="run-1",
        agent_id="agent-1",
        scenario_id="scenario-a",
        passed=True,
        score=0.8,
        registry=registry,
    )
    assert obs.mode == OnlineEvaluationMode.SHADOW
    assert registry.list_observations() == [obs]
    assert not hasattr(registry, "activate_profile")
    assert not hasattr(registry, "grant_execution")


def test_r2_eval_02_promotion_consumer_requires_separate_gate_evidence() -> None:
    bundle = PromotionEvidenceBundle(
        agent_id="agent-1",
        agent_version="1.0.0",
        source_stage=PromotionStage.DEV,
        target_stage=PromotionStage.STAGING,
        certification=AgentCertificationEvaluation(
            agent_id="agent-1",
            agent_version="1.0.0",
            eligible=False,
            reasons=["not ready"],
        ),
        evaluation_report_refs=[],
        rollback_plan_ref="",
        change_ticket_ref="",
    )
    decision = evaluate_agent_promotion(bundle)
    assert decision.approved is False
    assert any("evaluation report" in reason.lower() for reason in decision.reasons)


def test_r2_eval_03_registry_has_no_execution_activation_api() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    public = {name for name in dir(registry) if not name.startswith("_")}
    forbidden = {"activate", "grant_execution", "execute", "promote", "apply_profile"}
    assert forbidden.isdisjoint(public)


def test_r2_eval_04_decision_verification_distinct_from_evaluation_registry() -> None:
    dvc_source = inspect.getsource(build_decision_verification_pipeline)
    assert "OnlineEvaluationRegistry" not in dvc_source
    spec = DecisionVerificationPipelineBuildSpec()
    pipeline = build_decision_verification_pipeline(spec)
    assert not hasattr(pipeline, "append_observation")


def test_r2_eval_05_tenant_verdict_not_tenant_aware_at_this_stage() -> None:
    from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationObservation

    field_names = OnlineEvaluationObservation.model_fields.keys()
    assert "tenant_id" not in field_names
    registry = InMemoryOnlineEvaluationRegistry()
    record_shadow_observation(
        run_id="run-a",
        agent_id="agent-1",
        scenario_id="scenario-a",
        passed=True,
        score=1.0,
        registry=registry,
    )
    # run_id scoping is host/run identity only — not tenant isolation (TRACKED TENANT-X debt).
    assert all(obs.run_id == "run-a" for obs in registry.list_observations())


def test_r2_eval_06_shadow_mode_remains_non_authoritative() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    obs = record_shadow_observation(
        run_id="run-1",
        agent_id="agent-1",
        scenario_id="scenario-a",
        passed=True,
        score=1.0,
        registry=registry,
    )
    assert obs.mode == OnlineEvaluationMode.SHADOW
