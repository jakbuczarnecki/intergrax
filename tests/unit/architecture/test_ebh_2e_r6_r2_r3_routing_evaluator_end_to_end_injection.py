# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R2-R3 — routing evaluator end-to-end injection through resolver composition."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.llm.messages import ChatMessage
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
from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter as runtime_resolve_llm_adapter
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_RESOLVER = _REPO_ROOT / "intergrax/runtime/wiring/llm_resolver.py"
_APPLICATION_RESOLVER = _REPO_ROOT / "intergrax/applications/_shared/llm_resolver.py"


class _SequentialRoutingEvaluator:
    """Stateful external evaluator — no platform evaluator inheritance."""

    def __init__(
        self,
        *,
        first: LLMProfile,
        second: LLMProfile,
    ) -> None:
        self.calls = 0
        self.contexts: list[RoutingContext] = []
        self._first = first
        self._second = second

    def evaluate(
        self,
        profile: LLMRoutingProfile,
        context: RoutingContext,
    ) -> RoutingEvaluation:
        self.calls += 1
        self.contexts.append(context)
        selected = self._first if self.calls == 1 else self._second
        return RoutingEvaluation(
            selected_profile=selected,
            matched_rule_id=f"external-{self.calls}",
            routing_reason=f"external_call_{self.calls}",
            policy_route_hint=None,
            target=RoutingTarget(profile=selected, reason=f"external_call_{self.calls}"),
        )


def _routing_env() -> tuple[ApplicationEnvironmentProfile, LLMProfile, LLMProfile, LLMProfile]:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    profile_b = LLMProfile(provider=LLMProvider.VLLM, model="injected-profile-b")
    profile_c = LLMProfile(provider=LLMProvider.VLLM, model="injected-profile-c")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, profile_b, profile_c),
        rules=(BudgetBelowRule(threshold=0.25, profile=profile_b),),
    )
    return env, primary, profile_b, profile_c


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    return ""


def test_ebh_2e_r6_r2_r3_runtime_resolver_propagates_routing_evaluator_param() -> None:
    for name in ("resolve_llm_adapter", "_resolve_llm_adapter_impl"):
        source = _function_source(_RUNTIME_RESOLVER, name)
        assert "routing_evaluator" in source
    impl = _function_source(_RUNTIME_RESOLVER, "_resolve_llm_adapter_impl")
    assert "routing_evaluator=routing_evaluator" in impl


def test_ebh_2e_r6_r2_r3_application_resolver_propagates_routing_evaluator_param() -> None:
    for name in ("resolve_llm_adapter", "_resolve_llm_adapter_impl"):
        source = _function_source(_APPLICATION_RESOLVER, name)
        assert "routing_evaluator" in source
    impl = _function_source(_APPLICATION_RESOLVER, "_resolve_llm_adapter_impl")
    assert "routing_evaluator=routing_evaluator" in impl
    assert "wrap_routing_evaluating_adapter" in impl
    assert impl.count("routing_evaluator=routing_evaluator") >= 2


def test_ebh_2e_r6_r2_r3_public_resolver_signatures_accept_routing_evaluator() -> None:
    from intergrax.applications._shared import llm_resolver as app_resolver
    from intergrax.runtime.wiring import llm_resolver as runtime_resolver

    for fn in (
        runtime_resolver.resolve_llm_adapter,
        runtime_resolver.resolve_optional_llm_adapter,
        runtime_resolver._resolve_llm_adapter_impl,
        app_resolver.resolve_llm_adapter,
        app_resolver.resolve_optional_llm_adapter,
        app_resolver._resolve_llm_adapter_impl,
        app_resolver.resolve_environment_llm_adapter,
        app_resolver.resolve_optional_environment_llm_adapter,
    ):
        assert "routing_evaluator" in inspect.signature(fn).parameters


@pytest.mark.unit
@pytest.mark.gate
def test_ebh_2e_r6_r2_r3_runtime_public_resolver_uses_injected_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    profile_b = LLMProfile(provider=LLMProvider.VLLM, model="injected-profile-b")
    profile_c = LLMProfile(provider=LLMProvider.VLLM, model="injected-profile-c")
    env = RuntimeEnvironmentProfile(
        llm_profile=primary,
        llm_routing_profile=LLMRoutingProfile(
            default_profile=primary,
            allowed_profiles=(primary, profile_b, profile_c),
            rules=(BudgetBelowRule(threshold=0.25, profile=profile_b),),
        ),
    )
    evaluator = _SequentialRoutingEvaluator(first=profile_b, second=profile_c)

    def _fake_create(profile: LLMProfile, **_kwargs: object) -> FakeLLMAdapter:
        inner = FakeLLMAdapter(fixed_text="ok")
        inner.model = profile.model
        inner.provider = profile.provider
        return inner

    monkeypatch.setattr(
        "intergrax.runtime.wiring.llm_resolver.create_adapter",
        _fake_create,
    )
    monkeypatch.setattr(
        "intergrax.runtime.wiring.llm_resolver.create_adapter_with_failover",
        lambda profile, **_kwargs: _fake_create(profile),
    )

    adapter = runtime_resolve_llm_adapter(env, routing_evaluator=evaluator)
    assert evaluator.calls == 1
    assert adapter.model == "injected-profile-b"
    assert isinstance(evaluator, RoutingEvaluator)


@pytest.mark.unit
@pytest.mark.gate
def test_ebh_2e_r6_r2_r3_application_public_resolver_initial_and_live_share_injected_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _primary, profile_b, profile_c = _routing_env()
    evaluator = _SequentialRoutingEvaluator(first=profile_b, second=profile_c)

    def _fake_for_profile(profile: LLMProfile) -> FakeLLMAdapter:
        inner = FakeLLMAdapter(fixed_text="ok")
        inner.model = profile.model
        inner.provider = profile.provider
        return inner

    def _fake_create_for_routing(
        _env: object,
        evaluation: RoutingEvaluation,
        _ctx: object | None = None,
    ) -> FakeLLMAdapter:
        return _fake_for_profile(evaluation.selected_profile)

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.create_adapter_for_routing_evaluation",
        _fake_create_for_routing,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver._create_base_llm_adapter",
        lambda _env, profile, hint=None: _fake_for_profile(profile),
    )

    adapter = resolve_llm_adapter(env, routing_evaluator=evaluator)
    assert isinstance(adapter, RoutingEvaluatingLLMAdapter)
    assert evaluator.calls == 1
    assert adapter.model == "injected-profile-b"

    adapter.generate_messages([ChatMessage(role="user", content="live")])
    assert evaluator.calls == 2
    assert adapter.model == "injected-profile-c"
