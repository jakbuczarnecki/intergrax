# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Generator

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.instrumentation_span_attributes import (
    INTERGRAX_ATTEMPT_ID_ATTR,
    INTERGRAX_EXECUTION_ID_ATTR,
    INTERGRAX_RUN_ID_ATTR,
)
from intergrax.context.tracking.context_spans import (
    context_span,
    is_ce_otel_spans_enabled,
    set_ce_otel_spans_enabled,
)

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _enable_context_spans() -> Generator[None, None, None]:
    set_ce_otel_spans_enabled(True)
    yield
    set_ce_otel_spans_enabled(False)


def test_context_span_correlates_active_execution_identity(span_exporter: object) -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        with context_span("context.engine.assemble"):
            pass
    finally:
        reset_active_execution_identity(token)

    span = span_exporter.get_finished_spans()[0]
    attributes = dict(span.attributes)
    assert attributes[INTERGRAX_RUN_ID_ATTR] == str(run_id)
    assert attributes[INTERGRAX_ATTEMPT_ID_ATTR] == str(attempt_id)
    assert attributes[INTERGRAX_EXECUTION_ID_ATTR] == str(execution_id)


def test_context_span_instrumentation_failure_does_not_break_business(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_get_tracer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("otel provider unavailable")

    monkeypatch.setattr("opentelemetry.trace.get_tracer", _broken_get_tracer)

    outcome = "ok"
    with context_span("context.engine.assemble"):
        outcome = "still-ok"
    assert outcome == "still-ok"


def test_context_span_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_CE_OTEL_SPANS_ENABLED", "false")
    from intergrax.context.tracking import context_spans

    context_spans._ce_otel_spans_enabled_override = None
    assert is_ce_otel_spans_enabled() is False
    with context_span("context.engine.assemble"):
        pass
