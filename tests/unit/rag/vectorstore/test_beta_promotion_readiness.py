# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store
from intergrax.integrations.providers.vector_store.pinecone.bundle import create_pinecone_vector_store
from intergrax.rag.vectorstore.soak.prod_slo import (
    SoakConfig,
    evaluate_beta_promotion_readiness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_beta_promotion_readiness_passes_for_soaked_adapter() -> None:
    readiness = evaluate_beta_promotion_readiness(
        "pinecone",
        factory=create_pinecone_vector_store,
        config=SoakConfig(document_count=10, query_rounds=2, top_k=3, max_p95_query_ms=500.0),
    )
    assert readiness.manifest_beta is True
    assert readiness.harness_soak_passed is True
    assert readiness.ready is True
    assert readiness.reason == "harness_ready_pending_ops_live_soak"


def test_beta_promotion_rejects_non_candidate() -> None:
    readiness = evaluate_beta_promotion_readiness(
        "qdrant",
        factory=create_milvus_vector_store,
    )
    assert readiness.ready is False
    assert readiness.reason == "not_beta_candidate"
