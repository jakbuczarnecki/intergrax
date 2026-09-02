# © Artur Czarnecki. All rights reserved.

"""Bounded production routing observation for model-routing qualification."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.routing_evaluating_adapter import (
    RoutingContextProvider,
    RoutingEvaluatingLLMAdapter,
    RoutingEvaluationObserver,
)
from intergrax.llm_adapters.routing import RoutingEvaluation
from model_routing_qualifier.qualification_types import ObservedRoutingDecision


@dataclass(frozen=True, slots=True)
class RoutingObservationSession:
    previous_observer: RoutingEvaluationObserver | None
    previous_context_provider: RoutingContextProvider


def begin_routing_observation(
    evaluating: RoutingEvaluatingLLMAdapter,
    *,
    context_provider: RoutingContextProvider,
    captured: list[ObservedRoutingDecision],
) -> RoutingObservationSession:
    previous_observer = evaluating.on_evaluated_observer
    previous_context_provider = evaluating.context_provider

    def chained_observer(evaluation: RoutingEvaluation) -> None:
        captured.append(ObservedRoutingDecision.from_evaluation(evaluation))
        if previous_observer is not None:
            previous_observer(evaluation)

    evaluating.set_context_provider(context_provider)
    evaluating.set_on_evaluated(chained_observer)
    return RoutingObservationSession(
        previous_observer=previous_observer,
        previous_context_provider=previous_context_provider,
    )


def end_routing_observation(
    evaluating: RoutingEvaluatingLLMAdapter,
    session: RoutingObservationSession,
) -> None:
    evaluating.set_on_evaluated(session.previous_observer)
    evaluating.set_context_provider(session.previous_context_provider)


__all__ = [
    "begin_routing_observation",
    "end_routing_observation",
    "RoutingObservationSession",
]
