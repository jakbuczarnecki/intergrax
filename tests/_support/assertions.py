# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


def _trace_events(answer: RuntimeAnswer) -> List[Dict[str, Any]]:
    dt = answer.debug_trace or {}
    ev = dt.get("events", [])
    if not isinstance(ev, list):
        raise AssertionError("debug_trace['events'] is not a list")
    return ev  # type: ignore[return-value]


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
        if e.get("step") == step:
            return
    raise AssertionError(f"Trace does not contain step={step!r}. Steps={sorted({e.get('step') for e in events})}")


def assert_trace_has_run_start_end(answer: RuntimeAnswer) -> None:
    assert_trace_has_step(answer, step="run_start")
    assert_trace_has_step(answer, step="run_end")


def get_trace_event(answer: RuntimeAnswer, *, step: str, component: Optional[str] = None) -> Dict[str, Any]:
    events = _trace_events(answer)
    for e in events:
        if e.get("step") != step:
            continue
        if component is not None and e.get("component") != component:
            continue
        return e
    raise AssertionError(f"Trace event not found: step={step!r}, component={component!r}")
