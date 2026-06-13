# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.tracking.metrics import is_rag_metrics_enabled, set_rag_metrics_enabled
from intergrax.rag.tracking.rag_spans import set_rag_otel_spans_enabled

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_overrides() -> None:
    set_rag_metrics_enabled(None)
    set_rag_otel_spans_enabled(None)
    yield
    set_rag_metrics_enabled(None)
    set_rag_otel_spans_enabled(None)


def test_rag_metrics_default_on_when_otel_spans_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_RAG_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("INTERGRAX_RAG_OTEL_SPANS_ENABLED", raising=False)
    assert is_rag_metrics_enabled() is True


def test_rag_metrics_explicit_off_overrides_otel_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_RAG_METRICS_ENABLED", "false")
    monkeypatch.delenv("INTERGRAX_RAG_OTEL_SPANS_ENABLED", raising=False)
    assert is_rag_metrics_enabled() is False
