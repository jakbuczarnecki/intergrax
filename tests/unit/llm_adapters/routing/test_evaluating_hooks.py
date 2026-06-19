# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.routing.evaluating_hooks import (
    RoutingEvaluatingAdapterHooks,
    is_routing_evaluating_adapter,
    wire_routing_evaluating_hooks,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
)
from testing_support.builder import FakeLLMAdapter


class _HookedEvaluatingAdapter(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._inner = FakeLLMAdapter()
        self._on_evaluated = None
        self._on_allowlist_violation = None
        self._on_inner_swapped = None

    @property
    def inner_adapter(self) -> FakeLLMAdapter:
        return self._inner

    def set_on_evaluated(self, observer):  # type: ignore[no-untyped-def]
        self._on_evaluated = observer

    def set_on_allowlist_violation(self, observer):  # type: ignore[no-untyped-def]
        self._on_allowlist_violation = observer

    def set_on_inner_swapped(self, observer):  # type: ignore[no-untyped-def]
        self._on_inner_swapped = observer


@pytest.mark.unit
@pytest.mark.gate
def test_is_routing_evaluating_adapter_duck_types_wrapper() -> None:
    wrapped = _HookedEvaluatingAdapter()
    plain = FakeLLMAdapter()
    assert is_routing_evaluating_adapter(wrapped)
    assert isinstance(wrapped, RoutingEvaluatingAdapterHooks)
    assert not is_routing_evaluating_adapter(plain)


@pytest.mark.unit
@pytest.mark.gate
def test_wire_routing_evaluating_hooks_attaches_observers() -> None:
    wrapped = _HookedEvaluatingAdapter()
    plain = FakeLLMAdapter()
    observed: list[str] = []
    failover: list[object] = []

    wired = wire_routing_evaluating_hooks(
        wrapped,
        on_evaluated=lambda _evaluation: observed.append("evaluated"),
        on_allowlist_violation=lambda _exc, _ctx: observed.append("allowlist"),
        on_inner_swapped=lambda _inner: observed.append("swapped"),
        attach_failover_observer=lambda adapter: failover.append(adapter),
    )
    assert wired is True
    assert failover == [wrapped.inner_adapter]

    evaluation = LLMRoutingEvaluator().evaluate(
        LLMRoutingProfile(
            default_profile=LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
            allowed_profiles=(LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),),
        ),
        RoutingContext(),
    )
    wrapped.set_on_evaluated(lambda ev: observed.append("callback"))
    wrapped._on_evaluated(evaluation)  # type: ignore[misc]
    assert "callback" in observed

    not_wired = wire_routing_evaluating_hooks(
        plain,
        on_evaluated=lambda _evaluation: observed.append("plain"),
        on_allowlist_violation=lambda _exc, _ctx: None,
        on_inner_swapped=lambda _inner: None,
        attach_failover_observer=lambda adapter: failover.append(adapter),
    )
    assert not_wired is False
    assert failover[-1] is plain
