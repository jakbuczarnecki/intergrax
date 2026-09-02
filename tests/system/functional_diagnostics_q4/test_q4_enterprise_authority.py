# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q4-R1 enterprise routing authority gates."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    RoutingContext,
    RoutingEvaluation,
)
from model_routing_qualifier.model_routing import artifact_ref_for_profile
from model_routing_qualifier.qualification_types import ObservedRoutingDecision
from model_routing_qualifier.routing_observation import (
    begin_routing_observation,
    end_routing_observation,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_PATH = _REPO_ROOT / "agents" / "model_routing_qualifier" / "steps" / "model_routing_job.py"
_Q4_AGENT_ROOT = _REPO_ROOT / "agents" / "model_routing_qualifier"
_Q4_TEST_ROOT = _REPO_ROOT / "tests" / "system" / "functional_diagnostics_q4"


def _job_ast() -> ast.Module:
    return ast.parse(_JOB_PATH.read_text(encoding="utf-8"))


def test_model_routing_job_does_not_call_llm_routing_evaluator() -> None:
    tree = _job_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.llm_adapters.routing":
            for alias in node.names:
                assert alias.name != "LLMRoutingEvaluator"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "evaluate":
                if isinstance(func.value, ast.Name):
                    assert func.value.id != "LLMRoutingEvaluator"
                if isinstance(func.value, ast.Call):
                    assert not (
                        isinstance(func.value.func, ast.Name)
                        and func.value.func.id == "LLMRoutingEvaluator"
                    )


def test_model_routing_job_uses_routing_observation_helpers() -> None:
    source = _JOB_PATH.read_text(encoding="utf-8")
    assert "begin_routing_observation" in source
    assert "end_routing_observation" in source


def test_observer_is_passive_for_same_context() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    inner_primary = FakeLLMAdapter(fixed_text="primary")
    inner_primary.model = "gpt-4o-mini"
    inner_local = FakeLLMAdapter(fixed_text="local")
    inner_local.model = "meta-llama/Llama-3.1-8B"

    def _factory(evaluation: RoutingEvaluation, _ctx: RoutingContext) -> FakeLLMAdapter:
        if evaluation.selected_profile.model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    context = RoutingContext(budget_remaining_ratio=0.1)
    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner_primary,
        context_provider=lambda: context,
        adapter_factory=_factory,
    )

    adapter.generate_messages([ChatMessage(role="user", content="without")], run_id="without")
    without_ref = artifact_ref_for_profile(local)

    captured: list[ObservedRoutingDecision] = []
    session = begin_routing_observation(
        adapter,
        context_provider=lambda: context,
        captured=captured,
    )
    try:
        adapter.generate_messages([ChatMessage(role="user", content="with")], run_id="with")
    finally:
        end_routing_observation(adapter, session)

    assert captured[-1].selected_profile_ref == without_ref


def test_observer_cleanup_restores_previous_callback() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary,),
        rules=(),
    )
    inner = FakeLLMAdapter(fixed_text="ok")
    inner.model = "gpt-4o-mini"
    observed: list[str] = []

    def previous_observer(_evaluation: RoutingEvaluation) -> None:
        observed.append("previous")

    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner,
        context_provider=lambda: RoutingContext(),
        adapter_factory=lambda _evaluation, _ctx: inner,
        on_evaluated=previous_observer,
    )
    captured: list[ObservedRoutingDecision] = []
    session = begin_routing_observation(
        adapter,
        context_provider=lambda: RoutingContext(),
        captured=captured,
    )
    end_routing_observation(adapter, session)
    adapter.generate_messages([ChatMessage(role="user", content="after-restore")], run_id="restore")
    assert observed == ["previous"]
    assert captured == []


def test_concurrent_observations_do_not_cross_capture_buffers() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    inner_primary = FakeLLMAdapter(fixed_text="primary")
    inner_primary.model = "gpt-4o-mini"
    inner_local = FakeLLMAdapter(fixed_text="local")
    inner_local.model = "meta-llama/Llama-3.1-8B"

    def _factory(evaluation: RoutingEvaluation, _ctx: RoutingContext) -> FakeLLMAdapter:
        if evaluation.selected_profile.model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    ratios = {"run-a": 0.9, "run-b": 0.1}
    captures: dict[str, list[ObservedRoutingDecision]] = {"run-a": [], "run-b": []}
    lock = threading.Lock()

    def _run(run_id: str) -> None:
        ratio = ratios[run_id]
        adapter = RoutingEvaluatingLLMAdapter(
            env=env,
            inner=inner_primary,
            context_provider=lambda r=ratio: RoutingContext(budget_remaining_ratio=r),
            adapter_factory=_factory,
        )
        session = begin_routing_observation(
            adapter,
            context_provider=lambda r=ratio: RoutingContext(budget_remaining_ratio=r),
            captured=captures[run_id],
        )
        try:
            adapter.generate_messages(
                [ChatMessage(role="user", content=f"call-{run_id}")],
                run_id=run_id,
            )
        finally:
            end_routing_observation(adapter, session)

    threads = [
        threading.Thread(target=_run, args=("run-a",)),
        threading.Thread(target=_run, args=("run-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with lock:
        assert captures["run-a"][-1].model == "gpt-4o-mini"
        assert captures["run-b"][-1].model == "meta-llama/Llama-3.1-8B"


def test_oracle_expected_profile_not_in_qualification_job() -> None:
    source = _JOB_PATH.read_text(encoding="utf-8")
    assert "functional_diagnostics_q4" not in source
    assert "PROFILE_A_REF" not in source


@pytest.mark.parametrize(
    "path",
    [
        _JOB_PATH,
        _Q4_AGENT_ROOT / "qualification_types.py",
        _Q4_AGENT_ROOT / "routing_observation.py",
        _Q4_AGENT_ROOT / "diagnostics.py",
        _Q4_AGENT_ROOT / "model_routing_functional_evidence.py",
    ],
)
def test_q4_r1_files_avoid_forbidden_typing_tokens(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                assert alias.name != "Any"
        if isinstance(node, ast.Attribute) and node.attr in {"getattr", "setattr", "hasattr"}:
            raise AssertionError(f"reflection attribute access in {path}")
        if isinstance(node, ast.Name) and node.id == "inspect":
            raise AssertionError(f"inspect usage in {path}")
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                elements = slice_node.elts
                if len(elements) == 2:
                    if (
                        isinstance(elements[0], ast.Name)
                        and elements[0].id == "dict"
                        and isinstance(elements[1], ast.Name)
                        and elements[1].id in {"Any", "object"}
                    ):
                        raise AssertionError(f"forbidden dict annotation in {path}")
