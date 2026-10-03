# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R1 closed-world Evaluation plane proofs (CX-04)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EvaluationProfile,
)
from intergrax.runtime.architecture.online_evaluation import record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationMode
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.token_optimization.advisory_evaluation import (
    TokenOptimizationAdvisoryEvaluationCase,
    TokenOptimizationAdvisoryEvaluationResult,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_r1_eval_01_canonical_evaluation_profile_wiring() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctrlx.eval")
    wiring = wire_application_evaluation(env)
    assert wiring.profile.online_registry_enabled is True
    assert wiring.registry is not None


def test_r1_eval_02_online_evaluation_registry_append_read() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    record_shadow_observation(
        run_id="run-1",
        agent_id="agent-1",
        scenario_id="scenario-a",
        passed=True,
        score=0.9,
        registry=registry,
    )
    assert len(registry.list_observations()) == 1


def test_r1_eval_03_shadow_observation_is_observation_not_permission() -> None:
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


def test_r1_eval_04_advisory_evaluation_rejects_auto_apply() -> None:
    with pytest.raises(ValueError, match="auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryEvaluationResult(
            case_id="case-1",
            passed=True,
            actual_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            actual_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            actual_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            auto_apply_allowed=True,
            raw_content_included=False,
            recommendation_source_type=TokenOptimizationSourceType.PROMPT,
        )


def test_r1_eval_04b_offline_case_rejects_auto_apply_expectation() -> None:
    signal = TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.PROMPT,
        strategy_kind=TokenOptimizationStrategyKind.LOSSLESS_NORMALIZATION,
    )
    with pytest.raises(ValueError, match="expected_auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryEvaluationCase(
            case_id="case-1",
            title="t",
            signal=signal,
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            expected_auto_apply_allowed=True,
        )


def test_r1_eval_05_evaluation_profile_does_not_grant_execution_fields() -> None:
    profile = EvaluationProfile(shadow_eval_enabled=True)
    assert not hasattr(profile, "grant_execution")
    assert not hasattr(profile, "activate_profile")


def test_r1_eval_06_decision_verification_module_distinct_from_evaluation_registry() -> None:
    from intergrax.runtime import decision_verification_composition as dvc

    assert "OnlineEvaluationRegistry" not in dvc.__doc__ if dvc.__doc__ else True
    assert not hasattr(dvc, "append_observation")


def test_r1_eval_07_evaluation_observation_host_scoped_not_cross_run() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    record_shadow_observation(
        run_id="run-tenant-a",
        agent_id="agent-1",
        scenario_id="scenario-a",
        passed=True,
        score=1.0,
        registry=registry,
    )
    stored = registry.list_observations()
    assert all(obs.run_id == "run-tenant-a" for obs in stored)
