# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    RoutingContext,
)
from testing_support.builder import FakeLLMAdapter


@pytest.mark.integration
@pytest.mark.gate
def test_parallel_routing_runs_do_not_share_adapter_state(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _fake_create(
        _env: object,
        evaluation: object,
        _ctx: object | None = None,
    ) -> FakeLLMAdapter:
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        assert isinstance(evaluation, RoutingEvaluation)
        if evaluation.selected_profile.model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.create_adapter_for_routing_evaluation",
        _fake_create,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver._create_base_llm_adapter",
        lambda _env, _profile, hint=None: inner_primary,
    )

    ratios = {
        "run-a": {"ratio": 0.9},
        "run-b": {"ratio": 0.1},
    }
    models_seen: dict[str, str] = {}
    lock = threading.Lock()

    def _run(run_id: str) -> None:
        holder = ratios[run_id]
        adapter = resolve_llm_adapter(
            env,
            routing_context=RoutingContext(budget_remaining_ratio=holder["ratio"]),
            context_provider=lambda h=holder: RoutingContext(budget_remaining_ratio=h["ratio"]),
        )
        assert isinstance(adapter, RoutingEvaluatingLLMAdapter)
        adapter.generate_messages(
            [ChatMessage(role="user", content=f"call-{run_id}")],
            run_id=run_id,
        )
        with lock:
            models_seen[run_id] = adapter.model

    threads = [
        threading.Thread(target=_run, args=("run-a",)),
        threading.Thread(target=_run, args=("run-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert models_seen["run-a"] == "gpt-4o-mini"
    assert models_seen["run-b"] == "meta-llama/Llama-3.1-8B"
