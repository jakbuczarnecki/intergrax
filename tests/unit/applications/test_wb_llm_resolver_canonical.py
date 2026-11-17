# © Artur Czarnecki. All rights reserved.

"""W-B Part A — the Tier-1 host LLM resolver is the one canonical resolver (NT-WB-02, 04-07, 27, 28)."""

from __future__ import annotations

import threading

import intergrax.runtime.wiring.llm_resolver as canonical
import pytest
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.registration_contract import LLMProviderNotConfiguredError
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext
from intergrax.llm_adapters.routing.composition import resolve_routing_evaluator
from intergrax.llm_adapters.routing.contracts import AllowlistViolationError, RoutingEvaluation
from intergrax.llm_adapters.routing.evaluating_adapter import RoutingEvaluatingLLMAdapter

pytestmark = [pytest.mark.unit]


class _StubAdapter(BaseLLMAdapter):
    provider = LLMProvider.OPENAI
    model: str = "stub"

    def __init__(self, *, model: str) -> None:
        super().__init__()
        self.model = model
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(self, messages: list[ChatMessage], **kwargs: object) -> LLMAdapterResponse:
        del messages, kwargs
        self.calls += 1
        return LLMAdapterResponse(
            content=f"ok-{self.model}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


def _profile(model: str) -> LLMProfile:
    return LLMProfile(provider=LLMProvider.OPENAI, model=model)


def _routing_env(model: str) -> RuntimeEnvironmentProfile:
    primary = _profile(model)
    other = _profile(f"{model}-low")
    return RuntimeEnvironmentProfile(
        llm_profile=primary,
        llm_routing_profile=LLMRoutingProfile(
            default_profile=primary,
            allowed_profiles=(primary, other),
            rules=(BudgetBelowRule(threshold=0.25, profile=other),),
        ),
    )


@pytest.fixture(autouse=True)
def _stub_adapter_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_create(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        return _StubAdapter(model=profile.model or "solo")

    def _fake_failover(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        return _StubAdapter(model=profile.model or "solo")

    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.create_adapter", _fake_create)
    monkeypatch.setattr(
        "intergrax.runtime.wiring.llm_resolver.create_adapter_with_failover",
        _fake_failover,
    )
    monkeypatch.setattr("intergrax.runtime.wiring.llm_resolver.llm_profile_from_env", lambda: None)


def _evaluation(model: str, tenant: str) -> RoutingEvaluation:
    env = _routing_env(model)
    assert env.llm_routing_profile is not None
    return resolve_routing_evaluator(None).evaluate(env.llm_routing_profile, RoutingContext(tenant_id=tenant))


def test_nt_wb_02_routing_record_is_per_adapter() -> None:
    assert not hasattr(canonical, "_last_routing_evaluation")
    results: dict[str, tuple[LLMAdapter, str]] = {}
    barrier = threading.Barrier(2)

    def _resolve(key: str, model: str, tenant: str) -> None:
        barrier.wait()
        adapter = canonical.resolve_llm_adapter(_routing_env(model), tenant_id=tenant)
        results[key] = (adapter, tenant)

    threads = [
        threading.Thread(target=_resolve, args=("a", "model-a", "tenant-a")),
        threading.Thread(target=_resolve, args=("b", "model-b", "tenant-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    adapter_a, tenant_a = results["a"]
    adapter_b, tenant_b = results["b"]
    record_a = canonical.consume_routing_evaluation(adapter_a, tenant_id=tenant_a)
    record_b = canonical.consume_routing_evaluation(adapter_b, tenant_id=tenant_b)
    assert record_a is not None and record_b is not None
    assert record_a.selected_profile.model == "model-a"
    assert record_b.selected_profile.model == "model-b"
    assert canonical.consume_routing_evaluation(adapter_a, tenant_id=tenant_a) is None


def test_nt_wb_04_required_features_mismatch_refused() -> None:
    env = RuntimeEnvironmentProfile(llm_profile=_profile("plain"))
    with pytest.raises(LLMCapabilityMismatchError):
        canonical.resolve_llm_adapter(
            env,
            required_llm_features=(LLMFeature.STRICT_TOOL_ARGUMENTS,),
        )


def test_nt_wb_05_routing_profile_wraps_adapter() -> None:
    evaluations: list[RoutingEvaluation] = []

    class _CountingEvaluator:
        def evaluate(self, profile: LLMRoutingProfile, context: RoutingContext) -> RoutingEvaluation:
            evaluation = resolve_routing_evaluator(None).evaluate(profile, context)
            evaluations.append(evaluation)
            return evaluation

    adapter = canonical.resolve_llm_adapter(
        _routing_env("wrapped"),
        routing_evaluator=_CountingEvaluator(),
    )
    assert isinstance(adapter, RoutingEvaluatingLLMAdapter)
    before = len(evaluations)
    adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert len(evaluations) == before + 1


def test_nt_wb_06_unconfigured_provider_refused() -> None:
    with pytest.raises(LLMProviderNotConfiguredError):
        canonical.resolve_llm_adapter(None)
    with pytest.raises(LLMProviderNotConfiguredError):
        canonical.resolve_llm_adapter(RuntimeEnvironmentProfile())


def test_nt_wb_07_allowlist_violation_propagates() -> None:
    class _Violating:
        def evaluate(self, profile: LLMRoutingProfile, context: RoutingContext) -> RoutingEvaluation:
            del profile, context
            raise AllowlistViolationError("outside allowlist")

    with pytest.raises(AllowlistViolationError):
        canonical.resolve_llm_adapter(_routing_env("violating"), routing_evaluator=_Violating())


def test_nt_wb_27_shared_adapter_record_is_tenant_scoped() -> None:
    adapter = _StubAdapter(model="shared")
    evaluation_a = _evaluation("tenant-a-model", "A")
    evaluation_b = _evaluation("tenant-b-model", "B")
    canonical.record_routing_evaluation(adapter, evaluation_a, tenant_id="A")
    canonical.record_routing_evaluation(adapter, evaluation_b, tenant_id="B")

    consumed_a = canonical.consume_routing_evaluation(adapter, tenant_id="A")
    consumed_b = canonical.consume_routing_evaluation(adapter, tenant_id="B")
    assert consumed_a is evaluation_a
    assert consumed_b is evaluation_b
    assert canonical.consume_routing_evaluation(adapter, tenant_id="A") is None


def test_nt_wb_28_foreign_tenant_consume_returns_nothing() -> None:
    adapter = _StubAdapter(model="shared")
    canonical.record_routing_evaluation(adapter, _evaluation("tenant-a-model", "A"), tenant_id="A")
    assert canonical.consume_routing_evaluation(adapter, tenant_id="B") is None
    assert canonical.consume_routing_evaluation(adapter, tenant_id=None) is None
    assert canonical.consume_routing_evaluation(adapter, tenant_id="A") is not None
