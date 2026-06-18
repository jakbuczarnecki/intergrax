# © Artur Czarnecki. All rights reserved.

"""CE-MAINT-01/02 maintenance tests."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import AssembledContext, ContextFragment, ContextFragmentSource
from intergrax.context.tracking.assembly_cost import assembly_cost_from_fragments
from intergrax.context.tracking.context_spans import CE_OTEL_SPAN_NAMES, context_span, is_ce_otel_spans_enabled
from intergrax.llm.messages import ChatMessage

pytestmark = pytest.mark.gate


def test_ce_otel_span_names_registered() -> None:
    assert "context.engine.assemble" in CE_OTEL_SPAN_NAMES


def test_context_span_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_CE_OTEL_SPANS_ENABLED", "false")
    from intergrax.context.tracking import context_spans

    context_spans._ce_otel_spans_enabled_override = None
    assert is_ce_otel_spans_enabled() is False
    with context_span("context.engine.assemble"):
        pass


def test_assembly_cost_sums_fragment_tokens() -> None:
    fragments = (
        ContextFragment(
            fragment_id="a",
            content="hello",
            source=ContextFragmentSource.RAG,
            source_id="rag-1",
            token_estimate=10,
            relevance_score=0.9,
            freshness_score=0.8,
            confidence_score=0.85,
            mandatory=False,
        ),
        ContextFragment(
            fragment_id="b",
            content="world",
            source=ContextFragmentSource.TOOL_OUTPUT,
            source_id="tool-1",
            token_estimate=5,
            relevance_score=0.7,
            freshness_score=0.9,
            confidence_score=0.8,
            mandatory=False,
        ),
    )
    cost = assembly_cost_from_fragments(fragments)
    assert cost.fragment_token_cost == 15
    assert cost.estimated_cost_microusd == 15

    assembled = AssembledContext(
        messages=(ChatMessage(role="user", content="hello world"),),
        fragments_included=fragments,
        fragments_excluded=(),
        provenance=(),
        total_tokens=15,
        budget_tokens=128,
        degradation_steps=0,
    )
    from intergrax.context.tracking.assembly_cost import assembly_cost_from_assembled

    assert assembly_cost_from_assembled(assembled).fragment_token_cost == 15
