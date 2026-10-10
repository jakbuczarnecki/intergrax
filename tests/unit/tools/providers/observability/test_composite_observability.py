# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composite observability backend resolution (Tier A harness)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.contracts.integration_profile import (
    IntegrationProfile,
    ObservabilityRoleBindings,
)
from intergrax.integrations.contracts.observability_backend import TraceQueryResult, TraceRecord
from intergrax.integrations.contracts.shipped_manifests import LANGFUSE, LANGSMITH, SENTRY
from intergrax.tools.providers.observability.contracts import ErrorsCaptureInput, TracesQueryInput
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.providers.observability.service import errors_capture, traces_query
from intergrax.tools.registry.wiring import ObservabilityRoleBackends, ToolWiringContext

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


class _LangfuseBackend:
    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        return TraceQueryResult(
            traces=[TraceRecord(trace_id="lf-1", name=name or "run")],
        )


class _BraintrustBackend:
    def log_eval(self, *, name: str, score: float, metadata: Any = None, project: Any = None) -> str:
        return "bt-1"


def _role_ctx(sentry: _SentryBackend, traces: Any) -> ToolWiringContext:
    return ToolWiringContext(
        observability_backend=sentry,
        observability_backends={"sentry": sentry, "langsmith": traces},
        observability_role_backends=ObservabilityRoleBackends(errors=sentry, traces=traces),
    )


def test_resolve_errors_and_traces_separately() -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    ctx = _role_ctx(sentry, langsmith)
    assert resolve_observability_backend(ctx, role="errors") is sentry
    assert resolve_observability_backend(ctx, role="traces") is langsmith


def test_errors_capture_uses_sentry_not_langsmith() -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    ctx = _role_ctx(sentry, langsmith)
    out = errors_capture(ctx, ErrorsCaptureInput(message="boom", level="error"))
    assert out.event_id == "sentry:boom:error"


def test_traces_query_uses_langsmith_not_sentry() -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    ctx = _role_ctx(sentry, langsmith)
    out = traces_query(ctx, TracesQueryInput(limit=5))
    assert out.traces[0].trace_id == "ls-1"


def test_harness_profile_options_resolve_observability_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()

    def _fake_resolve(category: Any, *, slug: Any = None, profile: Any = None, config: Any = None) -> Any:
        if slug == "langsmith":
            return langsmith
        return sentry

    def _fake_get_entry(slug: str) -> SimpleNamespace:
        if slug in {"langsmith", "sentry"}:
            return SimpleNamespace(categories=(IntegrationCategory.OBSERVABILITY_BACKEND,))
        raise UnknownIntegrationError(slug)

    monkeypatch.setattr(
        "intergrax.integrations.registry.factory.resolve",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "intergrax.integrations.registry.catalog.get_entry",
        _fake_get_entry,
    )
    ctx = ToolWiringContext.from_integration_profile(IntegrationProfile.harness_lab())
    assert "sentry" in ctx.observability_backends
    assert "langsmith" in ctx.observability_backends
    assert resolve_observability_backend(ctx, role="errors") is sentry
    assert resolve_observability_backend(ctx, role="traces") is langsmith
    assert resolve_observability_backend(ctx, role="default") is sentry


def test_default_role_requires_explicit_observability_backend() -> None:
    ctx = ToolWiringContext(
        observability_backends={"langsmith": _LangSmithBackend()},
    )
    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        resolve_observability_backend(ctx, role="default")


def test_explicit_traces_role_binding_ignores_vendor_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    langfuse = _LangfuseBackend()

    def _fake_resolve(category: Any, *, slug: Any = None, profile: Any = None, config: Any = None) -> Any:
        return {"sentry": sentry, "langsmith": langsmith, "langfuse": langfuse}[slug]

    def _fake_get_entry(slug: str) -> SimpleNamespace:
        if slug in {"langsmith", "langfuse", "sentry"}:
            return SimpleNamespace(categories=(IntegrationCategory.OBSERVABILITY_BACKEND,))
        raise UnknownIntegrationError(slug)

    monkeypatch.setattr("intergrax.integrations.registry.factory.resolve", _fake_resolve)
    monkeypatch.setattr("intergrax.integrations.registry.catalog.get_entry", _fake_get_entry)

    options = {LANGSMITH.slug: {}, LANGFUSE.slug: {}, SENTRY.slug: {}}
    for traces_ref, expected in ((LANGFUSE, langfuse), (LANGSMITH, langsmith)):
        profile = IntegrationProfile(
            observability_backend=SENTRY,
            observability_roles=ObservabilityRoleBindings(errors=SENTRY, traces=traces_ref),
            options=options,
        )
        ctx = ToolWiringContext.from_integration_profile(profile)
        assert resolve_observability_backend(ctx, role="traces") is expected


def test_traces_role_without_explicit_binding_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentry = _SentryBackend()
    langsmith = _LangSmithBackend()
    langfuse = _LangfuseBackend()

    def _fake_resolve(category: Any, *, slug: Any = None, profile: Any = None, config: Any = None) -> Any:
        return {"sentry": sentry, "langsmith": langsmith, "langfuse": langfuse}[slug]

    def _fake_get_entry(slug: str) -> SimpleNamespace:
        if slug in {"langsmith", "langfuse", "sentry"}:
            return SimpleNamespace(categories=(IntegrationCategory.OBSERVABILITY_BACKEND,))
        raise UnknownIntegrationError(slug)

    monkeypatch.setattr("intergrax.integrations.registry.factory.resolve", _fake_resolve)
    monkeypatch.setattr("intergrax.integrations.registry.catalog.get_entry", _fake_get_entry)

    profile = IntegrationProfile(
        observability_backend=SENTRY,
        options={LANGSMITH.slug: {}, LANGFUSE.slug: {}, SENTRY.slug: {}},
    )
    ctx = ToolWiringContext.from_integration_profile(profile)
    with pytest.raises(RuntimeError, match="observability_role_backend_not_configured:traces"):
        resolve_observability_backend(ctx, role="traces")


def test_no_role_fallback_when_default_backend_has_traces() -> None:
    sentry = _SentryBackend()
    ctx = ToolWiringContext(
        observability_backend=sentry,
        observability_backends={"sentry": sentry},
    )
    with pytest.raises(RuntimeError, match="observability_role_backend_not_configured:traces"):
        resolve_observability_backend(ctx, role="traces")


def test_resolve_structural_regression_no_slug_ranking() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[5]
    text = (repo_root / "intergrax/tools/providers/observability/resolve.py").read_text(
        encoding="utf-8",
    )
    for forbidden in (
        "_TRACES_SLUGS",
        "_ERRORS_SLUGS",
        "_LOGS_SLUGS",
        "_EVAL_SLUGS",
        "_sanctioned_slug_backend",
        "next(iter(",
    ):
        assert forbidden not in text
