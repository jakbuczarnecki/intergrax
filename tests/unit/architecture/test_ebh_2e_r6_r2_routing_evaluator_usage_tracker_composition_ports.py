# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R2 — routing evaluator & usage tracker composition ports."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.applications._shared.llm_resolver import evaluate_llm_routing
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.base.usage_log import LLMRunStats, LLMUsageTrackable
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.routing_evaluator import RoutingEvaluator
from intergrax.llm_adapters.contracts.routing_profile import (
    LLMRoutingProfile,
    RoutingContext,
    RoutingEvaluation,
    RoutingTarget,
)
from intergrax.llm_adapters.routing import BudgetBelowRule
from intergrax.llm_adapters.contracts.llm_usage_report import LLMUsageReport
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.wiring.llm_resolver import evaluate_llm_routing as runtime_evaluate_llm_routing
from testing_support.builder import FakeLLMAdapter, build_runtime_state_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_STATE_SOURCE = _REPO_ROOT / "intergrax/runtime/nexus/engine/runtime_state.py"
_EVALUATING_ADAPTER_SOURCE = (
    _REPO_ROOT / "intergrax/llm_adapters/routing/evaluating_adapter.py"
)


def test_ebh_2e_r6_r2_runtime_state_does_not_import_concrete_usage_tracker() -> None:
    text = _RUNTIME_STATE_SOURCE.read_text(encoding="utf-8")
    assert "LLMUsageTracker" not in text


def test_ebh_2e_r6_r2_evaluating_adapter_does_not_instantiate_default_evaluator() -> None:
    text = _EVALUATING_ADAPTER_SOURCE.read_text(encoding="utf-8")
    assert "LLMRoutingEvaluator()" not in text


class ExternalRoutingEvaluator:
    """Structural routing evaluator — no platform evaluator inheritance."""

    def evaluate(
        self,
        profile: LLMRoutingProfile,
        context: RoutingContext,
    ) -> RoutingEvaluation:
        forced = LLMProfile(provider=LLMProvider.VLLM, model="external-forced")
        return RoutingEvaluation(
            selected_profile=forced,
            matched_rule_id="external",
            routing_reason="external_evaluator",
            policy_route_hint=None,
            target=RoutingTarget(profile=forced, reason="external_evaluator"),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_ebh_2e_r6_r2_application_resolver_accepts_external_evaluator() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.25, profile=local),),
    )
    selected, _hint, reason = evaluate_llm_routing(
        env,
        routing_context=RoutingContext(budget_remaining_ratio=0.1),
        routing_evaluator=ExternalRoutingEvaluator(),
    )
    assert selected.model == "external-forced"
    assert reason == "external_evaluator"
    assert isinstance(ExternalRoutingEvaluator(), RoutingEvaluator)


@pytest.mark.unit
@pytest.mark.gate
def test_ebh_2e_r6_r2_runtime_resolver_accepts_external_evaluator() -> None:
    from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile

    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = RuntimeEnvironmentProfile(
        llm_profile=primary,
        llm_routing_profile=LLMRoutingProfile(
            default_profile=primary,
            allowed_profiles=(primary, local),
            rules=(BudgetBelowRule(threshold=0.25, profile=local),),
        ),
    )
    selected, _hint, reason = runtime_evaluate_llm_routing(
        env,
        routing_context=RoutingContext(budget_remaining_ratio=0.1),
        routing_evaluator=ExternalRoutingEvaluator(),
    )
    assert selected.model == "external-forced"
    assert reason == "external_evaluator"


@dataclass
class _ExternalRunStats:
    _stats: LLMRunStats

    def get_run_stats(self, run_id: str | None = None) -> LLMRunStats | None:
        return self._stats


class ExternalUsageTrackable:
    provider = LLMProvider.OPENAI
    model = "external-tracked"

    def __init__(self, run_id: str) -> None:
        self.usage = _ExternalRunStats(
            LLMRunStats(calls=1, input_tokens=3, output_tokens=5, total_tokens=8)
        )
        self._run_id = run_id


class ExternalUsageAggregator:
    """Structural usage aggregator — no LLMUsageTracker inheritance."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._labels: list[str] = []

    def register_adapter(
        self,
        trackable: LLMUsageTrackable,
        label: str | None = None,
    ) -> None:
        self._labels.append(label or "default")

    def build_report(self) -> LLMUsageReport:
        total = LLMRunStats(calls=1, input_tokens=3, output_tokens=5, total_tokens=8)
        return LLMUsageReport(
            run_id=self.run_id,
            total=total,
            entries=[],
            by_provider_model={},
            adapter_instance_ids={},
        )


@pytest.mark.unit
@pytest.mark.gate
def test_ebh_2e_r6_r2_runtime_state_injected_external_usage_tracker() -> None:
    state = build_runtime_state_for_tests(run_id="run-ext-tracker")
    external = ExternalUsageAggregator(run_id=state.run_id)
    expected_run_id = state.run_id
    state.llm_usage_tracker = external
    trackable = FakeLLMAdapter()
    state.llm_usage_tracker.register_adapter(trackable, label="core_adapter")
    report = state.llm_usage_tracker.build_report()
    assert report.run_id == expected_run_id
    assert report.total.total_tokens == 8
    assert external._labels == ["core_adapter"]


def test_ebh_2e_r6_r2_configure_llm_tracker_uses_composition_not_concrete() -> None:
    source = inspect.getsource(RuntimeState.configure_llm_tracker)
    assert "LLMUsageTracker" not in source
    assert "ensure_llm_usage_tracker_on_state" in source


def test_ebh_2e_r6_r2_routing_evaluator_contract_module_is_pure() -> None:
    path = _REPO_ROOT / "intergrax/llm_adapters/contracts/routing_evaluator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime")
            assert not node.module.startswith("intergrax.applications")
            assert "evaluator" not in (node.module or "")
