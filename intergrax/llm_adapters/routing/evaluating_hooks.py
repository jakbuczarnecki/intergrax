# © Artur Czarnecki. All rights reserved.

"""Duck-typed hooks for routing-evaluating LLM adapters (M-LLM-X.13.1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing.contracts import RoutingContext, RoutingEvaluation

RoutingEvaluationObserver = Callable[[RoutingEvaluation], None]
AllowlistViolationObserver = Callable[[object, RoutingContext], None]
InnerSwappedObserver = Callable[[LLMAdapter], None]


@runtime_checkable
class RoutingEvaluatingAdapterHooks(Protocol):
    """Tier-0/1 surface for evaluating wrappers without importing Tier-3 types."""

    @property
    def inner_adapter(self) -> LLMAdapter: ...

    def set_on_evaluated(self, observer: RoutingEvaluationObserver | None) -> None: ...

    def set_on_allowlist_violation(self, observer: AllowlistViolationObserver | None) -> None: ...

    def set_on_inner_swapped(self, observer: InnerSwappedObserver | None) -> None: ...


def is_routing_evaluating_adapter(adapter: LLMAdapter | None) -> bool:
    """Return True when ``adapter`` exposes evaluating-wrapper hook methods."""
    return isinstance(adapter, RoutingEvaluatingAdapterHooks)


def wire_routing_evaluating_hooks(
    adapter: LLMAdapter,
    *,
    on_evaluated: RoutingEvaluationObserver,
    on_allowlist_violation: AllowlistViolationObserver,
    on_inner_swapped: InnerSwappedObserver,
    attach_failover_observer: Callable[[LLMAdapter], None],
) -> bool:
    """
    Attach routing trace observers to an evaluating wrapper when present.

    Returns False when ``adapter`` is not a routing-evaluating wrapper.
    """
    if not is_routing_evaluating_adapter(adapter):
        attach_failover_observer(adapter)
        return False
    evaluating = adapter
    evaluating.set_on_evaluated(on_evaluated)
    evaluating.set_on_allowlist_violation(on_allowlist_violation)
    evaluating.set_on_inner_swapped(on_inner_swapped)
    attach_failover_observer(evaluating.inner_adapter)
    return True
