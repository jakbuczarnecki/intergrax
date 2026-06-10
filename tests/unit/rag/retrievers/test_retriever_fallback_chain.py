# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.retrievers.resilience.retriever_fallback import retriever_fallback_chain

pytestmark = pytest.mark.gate


def test_fallback_chain_from_fusion() -> None:
    chain = retriever_fallback_chain(
        "fusion",
        ("fusion", "hybrid", "vector_similarity", "mmr"),
    )
    assert chain == ["fusion", "hybrid", "vector_similarity"]


def test_fallback_chain_from_hybrid() -> None:
    chain = retriever_fallback_chain("hybrid", ("hybrid", "vector_similarity"))
    assert chain == ["hybrid", "vector_similarity"]


def test_fallback_chain_for_unknown_primary_appends_canonical_tail() -> None:
    chain = retriever_fallback_chain(
        "multiquery",
        ("multiquery", "hybrid", "vector_similarity"),
    )
    assert chain == ["multiquery", "hybrid", "vector_similarity"]
