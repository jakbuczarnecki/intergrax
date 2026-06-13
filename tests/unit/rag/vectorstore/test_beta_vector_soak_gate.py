# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store
from intergrax.integrations.providers.vector_store.pinecone.bundle import create_pinecone_vector_store
from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store
from intergrax.rag.vectorstore.soak.prod_slo import (
    BETA_PROMOTION_CANDIDATE_SLUGS,
    SoakConfig,
    run_beta_adapter_soak,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("slug,factory", [
    ("pinecone", create_pinecone_vector_store),
    ("milvus", create_milvus_vector_store),
    ("vespa", create_vespa_vector_store),
])
def test_beta_vector_adapter_soak_passes_with_injected_store(slug: str, factory: object) -> None:
    assert slug in BETA_PROMOTION_CANDIDATE_SLUGS
    result = run_beta_adapter_soak(
        factory,
        slug=slug,
        config=SoakConfig(
            document_count=15,
            query_rounds=3,
            top_k=4,
            max_p95_query_ms=500.0,
        ),
    )
    assert result.passed is True, result.reason
    assert result.reason == "ok"
