# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent


def _trace_events(answer: RuntimeAnswer) -> List[TraceEvent]:
    # New contract: typed trace events on the RuntimeAnswer.
    return list(answer.trace_events or [])


def assert_answer_non_empty(answer: RuntimeAnswer) -> None:
    assert answer is not None, "RuntimeAnswer is None"
    assert isinstance(answer.answer, str), "RuntimeAnswer.answer is not a str"
    assert answer.answer.strip(), "RuntimeAnswer.answer is empty/blank"


def assert_route_strategy(answer: RuntimeAnswer, *, expected: str) -> None:
    assert answer.route is not None, "RuntimeAnswer.route is None"
    assert answer.route.strategy == expected, f"route.strategy={answer.route.strategy!r}, expected={expected!r}"


def assert_trace_has_step(answer: RuntimeAnswer, *, step: str) -> None:
    events = _trace_events(answer)
    for e in events:
        if e.step == step:
            return

    raise AssertionError(
        f"Trace does not contain step={step!r}. Steps={sorted({e.step for e in events})}"
    )


def assert_trace_has_run_start_end(answer: RuntimeAnswer) -> None:
    assert_trace_has_step(answer, step="run_start")
    assert_trace_has_step(answer, step="run_end")


def get_trace_event(
    answer: RuntimeAnswer,
    *,
    step: str,
    component: Optional[TraceComponent] = None,
) -> TraceEvent:
    events = _trace_events(answer)
    for e in events:
        if e.step != step:
            continue
        if component is not None and e.component != component:
            continue
        return e

    raise AssertionError(f"Trace event not found: step={step!r}, component={component!r}")


def trace_steps(answer: RuntimeAnswer) -> List[str]:
    return [e.step for e in _trace_events(answer)]
