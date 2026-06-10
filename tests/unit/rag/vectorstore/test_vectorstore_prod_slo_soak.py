# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store
from intergrax.rag.vectorstore.soak.prod_slo import (
    BETA_PROMOTION_CANDIDATE_SLUGS,
    STABLE_PROD_SLO_SLUGS,
    SoakConfig,
    manifest_status_for_slug,
    run_vectorstore_soak,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_stable_slug_manifests_are_stable() -> None:
    for slug in STABLE_PROD_SLO_SLUGS:
        assert manifest_status_for_slug(slug) is IntegrationStatus.STABLE, slug


def test_beta_promotion_candidates_remain_beta() -> None:
    for slug in BETA_PROMOTION_CANDIDATE_SLUGS:
        assert manifest_status_for_slug(slug) is IntegrationStatus.BETA, slug


def test_inmemory_manifest_remains_beta_harness_only() -> None:
    assert manifest_status_for_slug("inmemory") is IntegrationStatus.BETA


def test_soak_contract_passes_on_inmemory_vector_store() -> None:
    store = create_inmemory_vector_store(tenant_id="soak-tenant")
    result = run_vectorstore_soak(
        store,
        slug="inmemory",
        config=SoakConfig(
            document_count=20,
            query_rounds=3,
            top_k=4,
            max_p95_query_ms=500.0,
        ),
    )
    assert result.passed is True
    assert result.reason == "ok"
    assert result.documents_indexed == 20
    assert result.queries_executed == 3
