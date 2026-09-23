# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R5 — Tier-1 failover chain ordering convergence with application/live."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.llm_resolver import _create_base_llm_adapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.failover_policy import FailoverCandidateNotAuthorizedError
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.model_router import ModelRouter
from intergrax.llm_adapters.routing import LLMRoutingProfile
from intergrax.llm_adapters.routing.evaluator import profile_identity
from intergrax.llm_adapters.registry.profile import create_adapter_with_failover
from intergrax.runtime.wiring.llm_resolver import _resolve_llm_adapter_impl, resolve_llm_adapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TIER1_RESOLVER = _REPO_ROOT / "intergrax/runtime/wiring/llm_resolver.py"


class _HttpStatusError(RuntimeError):
    status_code: int

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StubAdapter(BaseLLMAdapter):
    provider = LLMProvider.OPENAI
    model: str = "stub"

    def __init__(self, *, model: str, fail: bool = False, status_code: int = 429) -> None:
        super().__init__()
        self.model = model
        self._fail = fail
        self._status_code = status_code
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        del kwargs
        self.dispatched = True
        if self._fail:
            raise _HttpStatusError("provider error", status_code=self._status_code)
        return LLMAdapterResponse(
            content=f"ok-{self.model}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


def _profile(model: str, *, provider: LLMProvider = LLMProvider.OPENAI) -> LLMProfile:
    return LLMProfile(provider=provider, model=model)


def _chain_primary() -> LLMProfile:
    profile_b = _profile("B", provider=LLMProvider.GROQ)
    profile_c = _profile("C", provider=LLMProvider.VLLM)
    return _profile("A").model_copy(update={"fallback_profiles": (profile_b, profile_c)})


def _materialization_order(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    order: list[str] = []

    def _fake_create(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        order.append(profile.model or "default")
        fail = profile.model == "B"
        return _StubAdapter(model=profile.model or "default", fail=fail, status_code=429)

    monkeypatch.setattr(
        "intergrax.llm_adapters.registry.profile.create_adapter",
        _fake_create,
    )
    return order


def test_model_router_canonical_cheapest_order_is_b_c_a() -> None:
    primary = _chain_primary()
    router = ModelRouter.from_profiles(
        primary,
        fallbacks=primary.fallback_profiles,
        policy_route_hint="cheapest",
    )
    assert [profile.model for profile in router.ordered_profiles()] == ["B", "C", "A"]


@pytest.mark.parametrize(
    ("hint", "expected_models"),
    [
        (None, ["A", "B", "C"]),
        ("fastest", ["A", "B", "C"]),
        ("cheapest", ["B", "C", "A"]),
        ("quality", ["A", "C", "B"]),
        ("balanced", ["B", "A", "C"]),
    ],
)
def test_tier1_materializes_full_failover_chain(
    monkeypatch: pytest.MonkeyPatch,
    hint: str | None,
    expected_models: list[str],
) -> None:
    order = _materialization_order(monkeypatch)
    env = RuntimeEnvironmentProfile(llm_profile=_chain_primary())
    _resolve_llm_adapter_impl(env, policy_route_hint=hint)
    assert order == expected_models


@pytest.mark.parametrize(
    ("hint", "expected_models"),
    [
        (None, ["A", "B", "C"]),
        ("fastest", ["A", "B", "C"]),
        ("cheapest", ["B", "C", "A"]),
        ("quality", ["A", "C", "B"]),
        ("balanced", ["B", "A", "C"]),
    ],
)
def test_application_live_matches_tier1_materialization(
    monkeypatch: pytest.MonkeyPatch,
    hint: str | None,
    expected_models: list[str],
) -> None:
    tier1_order = _materialization_order(monkeypatch)
    env = RuntimeEnvironmentProfile(llm_profile=_chain_primary())
    _resolve_llm_adapter_impl(env, policy_route_hint=hint)
    assert tier1_order == expected_models

    app_order = _materialization_order(monkeypatch)
    app_env = ApplicationEnvironmentProfile.lab_defaults()
    _create_base_llm_adapter(app_env, _chain_primary(), hint=hint)
    assert app_order == expected_models
    assert app_order == tier1_order


def test_tier1_no_fallback_uses_single_adapter_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    failover_calls: list[LLMProfile] = []
    create_calls: list[str] = []

    def _fake_failover(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        failover_calls.append(profile)
        return _StubAdapter(model=profile.model or "solo")

    def _fake_create(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        create_calls.append(profile.model or "solo")
        return _StubAdapter(model=profile.model or "solo")

    monkeypatch.setattr(
        "intergrax.runtime.wiring.llm_resolver.create_adapter_with_failover",
        _fake_failover,
    )
    monkeypatch.setattr(
        "intergrax.runtime.wiring.llm_resolver.create_adapter",
        _fake_create,
    )
    env = RuntimeEnvironmentProfile(llm_profile=_profile("solo"))
    adapter = resolve_llm_adapter(env)
    assert isinstance(adapter, _StubAdapter)
    assert create_calls == ["solo"]
    assert failover_calls == []


def test_tier1_hint_without_fallback_uses_failover_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _materialization_order(monkeypatch)
    solo = _profile("solo")
    env = RuntimeEnvironmentProfile(llm_profile=solo)
    _resolve_llm_adapter_impl(env, policy_route_hint="cheapest")
    assert order == ["solo"]


def test_tier1_unauthorized_candidate_fails_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _materialization_order(monkeypatch)
    chain = _chain_primary()
    profile_b = chain.fallback_profiles[0]
    routing = LLMRoutingProfile(
        default_profile=chain,
        allowed_profiles=(chain, chain.fallback_profiles[1]),
    )
    env = RuntimeEnvironmentProfile(llm_profile=chain, llm_routing_profile=routing)
    with pytest.raises(FailoverCandidateNotAuthorizedError, match=profile_identity(profile_b)):
        _resolve_llm_adapter_impl(env, policy_route_hint="cheapest")
    assert order == []


def test_tier1_retryable_failover_executes_full_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    order = _materialization_order(monkeypatch)
    env = RuntimeEnvironmentProfile(llm_profile=_chain_primary())
    adapter = _resolve_llm_adapter_impl(env, policy_route_hint="cheapest")
    assert order == ["B", "C", "A"]
    assert isinstance(adapter, FailoverLLMAdapter)
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "ok-C"


def test_r5_tier1_resolver_does_not_pre_order_before_failover_factory() -> None:
    tree = ast.parse(_TIER1_RESOLVER.read_text(encoding="utf-8"))
    impl_source = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_llm_adapter_impl":
            impl_source = ast.get_source_segment(
                _TIER1_RESOLVER.read_text(encoding="utf-8"),
                node,
            ) or ""
            break
    assert "ordered_profiles()[0]" not in impl_source
    assert "ModelRouter" not in impl_source


def test_r5_create_adapter_with_failover_still_canonical_for_direct_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _materialization_order(monkeypatch)
    create_adapter_with_failover(_chain_primary(), policy_route_hint="cheapest")
    assert order == ["B", "C", "A"]
