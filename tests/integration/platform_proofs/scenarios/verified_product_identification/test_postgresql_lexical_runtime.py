"""Bounded PostgreSQL runtime qualification for BM25 lexical search."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    LexicalSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    postgres_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.retrieval.composition import (
    build_lexical_candidate_search,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    RelationalTargetId,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    drop_postgresql_schema,
    runtime_qualification_target_name,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.test_postgresql_exact_identifier_runtime import (
    _relational_record,
)

pytestmark = [pytest.mark.integration]


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured locally")
def test_postgresql_lexical_runtime_sn850x_query() -> None:
    schema_name = runtime_qualification_target_name("lexical")
    drop_postgresql_schema(schema_name)
    storage = PostgreSqlRelationalStorageAdapter.from_env(schema_name=schema_name)
    target = RelationalTargetId("vpi-products")
    storage.prepare_target(target)
    records = tuple(_relational_record(index, f"runtime-offer-{index}") for index in range(3))
    storage.write_batch(RelationalBatch(target=target, records=records))

    lexical = build_lexical_candidate_search(schema_name=schema_name)
    try:
        result = lexical.search(LexicalSearchQuery(query_text="MZ-V9P2T0BW", limit=3))
        assert result.failure is None
        assert result.candidates
        assert result.candidates[0].channel_score is not None
        assert result.candidates[0].channel_score.bm25_score > 0.0
    finally:
        drop_postgresql_schema(schema_name)
