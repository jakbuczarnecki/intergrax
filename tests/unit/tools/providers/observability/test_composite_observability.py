# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composite observability backend resolution (Tier A harness)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations.contracts.observability_backend import TraceQueryResult, TraceRecord
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.observability.contracts import ErrorsCaptureInput, TracesQueryInput
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.providers.observability.service import errors_capture, traces_query
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _SentryBackend:
    def capture_message(self, message: str, *, level: str) -> str:
        return f"sentry:{message}:{level}"

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        return TraceQueryResult()


class _LangSmithBackend:
    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        return TraceQueryResult(
            traces=[TraceRecord(trace_id="ls-1", name=name or "run")],
        )


class _BraintrustBackend:
    def log_eval(self, *, name: str, score: float, metadata: Any = None, project: Any = None) -> str:
        return "bt-1"


def test_resolve_errors_and_traces_separately() -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    ctx = ToolWiringContext(
        observability_backend=sentry,
        observability_backends={"sentry": sentry, "langsmith": langsmith},
    )
    assert resolve_observability_backend(ctx, role="errors") is sentry
    assert resolve_observability_backend(ctx, role="traces") is langsmith


def test_errors_capture_uses_sentry_not_langsmith() -> None:
    ctx = ToolWiringContext(
        observability_backend=_SentryBackend(),
        observability_backends={
            "sentry": _SentryBackend(),
            "langsmith": _LangSmithBackend(),
        },
    )
    out = errors_capture(ctx, ErrorsCaptureInput(message="boom", level="error"))
    assert out.event_id == "sentry:boom:error"


def test_traces_query_uses_langsmith_not_sentry() -> None:
    ctx = ToolWiringContext(
        observability_backend=_SentryBackend(),
        observability_backends={
            "sentry": _SentryBackend(),
            "langsmith": _LangSmithBackend(),
        },
    )
    out = traces_query(ctx, TracesQueryInput(limit=5))
    assert out.traces[0].trace_id == "ls-1"


def test_harness_profile_options_resolve_observability_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()

    def _fake_resolve(category: Any, *, slug: Any = None, profile: Any = None, config: Any = None) -> Any:
        if slug == "langsmith":
            return langsmith
        return sentry

    monkeypatch.setattr(
        "intergrax.integrations.registry.factory.resolve",
        _fake_resolve,
    )
    ctx = ToolWiringContext.from_integration_profile(IntegrationProfile.harness_lab())
    assert "sentry" in ctx.observability_backends
    assert "langsmith" in ctx.observability_backends
    assert resolve_observability_backend(ctx, role="errors") is sentry
    assert resolve_observability_backend(ctx, role="traces") is langsmith
